"""Run 1: test-like simulation, scale-free counts, learned candidate pruning, larger training set.

Why: on the leaderboard US/India scored ~0.951 vs 0.965 locally. The test has 5.75 S2/S3 records per
S1 (train: 4.67), i.e. ~40% of pool records match no S1 (train: 26%). Removing ~19% of train S1
entities turns their S2/S3 records into unmatched look-alikes and gives 5.77 records per S1 -- the
test's density. All training/validation below runs in that simulated world.

Stages:  retrieval (5 channels)  ->  pruner (cheap blocking-stage signals only)  ->  matcher (full
features on the pruned candidates)  ->  exclusive + threshold.  candidate_pairs.tsv = pruned set,
which is exactly what the matcher scores.

    python -m ber.cascade diag      # old sub1 model evaluated in the simulated world
    python -m ber.cascade pruner    # train pruner on sample A, report recall / candidates per S1
    python -m ber.cascade trainset  # pruned + full features for sample B
    python -m ber.cascade cv        # 2-fold grouped CV on B -> F0.5 + threshold
    python -m ber.cascade final     # fit matcher on all of B
    python -m ber.cascade test      # prune + score test, write outputs
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import decide
from .blocking import cand_path, query_mask
from .blocking_eval import key
from .experiment import make_model
from .features import (_group_counts, _num_feats, _street, assemble, context_features, load_strings,
                       pair_features)
from .io import load_gt_pairs, load_records
from .metrics import per_entity_f05, summarize
from .paths import OUTPUT_DIR, work

RUN = "run6"   # names this run's model / scores / output folder
SET = "run6"   # candidate set (with reverse retrieval), pruner and train sets
VIEW = True    # run3: shift-robust matcher features
# run2: deeper lists + native-script and empty-address typo channels (pruner trims them back)
KS = {"joint": 30, "addr": 20, "name": 10, "cgram": 10, "name_noaddr": 10, "native": 10, "cgram_noaddr": 10,
      "joint_rev": 2}  # run6: each S2/S3 record's 2 best S1 (reverse retrieval)
COMP_CH = ("joint", "addr", "name")


# ----------------------------------------------------------------------------- samples / simulation
def masks(rec):
    """A: pruner training (10%, the old dev sample). B: matcher training (30%, disjoint).
    R: S1 removed from the world (~19% of all S1) so their pool records become unmatched look-alikes."""
    s1 = rec.source.values == 1
    A = query_mask("train", rec, 0.1)
    u = np.random.default_rng(1).random(len(rec))
    rest = s1 & ~A
    B = rest & (u < 0.3 / 0.9)
    R = rest & ~B & (np.random.default_rng(2).random(len(rec)) < 0.188 / 0.6)
    return A, B, R


# ----------------------------------------------------------------------------- scale-free context
def world_stats(split, keep_s1, tag):
    """Per record: how common its name / compact name / address is (as rates per 1M records of the same
    country, S1 side over kept S1 only), and per pool record: best/second-best claim by any kept S1 per
    channel and how many kept S1 lists contain it (relative to the expected number)."""
    path = work(split, f"world_{tag}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    rec = load_records(split, ["source", "country"])
    country = rec.country.cat.codes.values.astype(np.int64)
    pool = rec.source.values > 1
    s1w = keep_s1.astype(np.float64)
    n_s1 = np.bincount(country, weights=s1w, minlength=8)
    n_pool = np.bincount(country, weights=pool, minlength=8)
    parts = sorted(work(split, "norm").glob("part-*.parquet"))
    out = {}
    for col in ["nc", "nk", "an"]:
        v = pa.concat_tables([pq.read_table(p, columns=[col]) for p in parts]).column(col)
        codes = pa.chunked_array(v).dictionary_encode().combine_chunks().indices.to_numpy().astype(np.int64)
        uniq, inv = np.unique(codes * 8 + country, return_inverse=True)
        s1c = np.bincount(inv, weights=s1w, minlength=len(uniq))[inv]
        pc = np.bincount(inv, weights=pool, minlength=len(uniq))[inv]
        empty = np.asarray(pa.compute.equal(v, "").to_numpy(zero_copy_only=False))
        r1 = np.log1p(s1c / np.maximum(n_s1[country], 1) * 1e6).astype(np.float32)
        r2 = np.log1p(pc / np.maximum(n_pool[country], 1) * 1e6).astype(np.float32)
        r1[empty] = np.nan; r2[empty] = np.nan
        out[f"{col}_s1r"], out[f"{col}_poolr"] = r1, r2
        out[f"{col}_s1n_raw"] = np.where(empty, np.nan, np.log1p(s1c)).astype(np.float32)
        out[f"{col}_pooln_raw"] = np.where(empty, np.nan, np.log1p(pc)).astype(np.float32)
    n = len(rec)
    for ch in COMP_CH:
        t = pq.read_table(cand_path(split, ch, 1.0), columns=["s1", "p", "score", "rank"], filters=[("rank", "<", KS[ch])])
        s1, p, s = t.column("s1").to_numpy(), t.column("p").to_numpy(), t.column("score").to_numpy()
        k = keep_s1[s1]
        p, s = p[k], s[k]
        o = np.lexsort((-s, p))
        p, s = p[o], s[o]
        first = np.r_[True, p[1:] != p[:-1]]
        second = np.r_[False, first[:-1]] & ~first
        m1 = np.zeros(n, np.float32); m2 = np.zeros(n, np.float32)
        m1[p[first]] = s[first]; m2[p[second]] = s[second]
        cnt = np.bincount(p, minlength=n).astype(np.float64)
        expected = KS[ch] * n_s1[country] / np.maximum(n_pool[country], 1)
        out[f"{ch}_max1"], out[f"{ch}_max2"] = m1, m2
        out[f"{ch}_nrel"] = np.log1p(cnt / expected).astype(np.float32)
        out[f"{ch}_n_raw"] = np.log1p(cnt).astype(np.float32)
    df = pd.DataFrame(out)
    df.to_parquet(path, index=False)
    return df


def world_features(pairs, W):
    a, b = pairs.s1.values, pairs.p.values
    f = {}
    for col in ["nc", "nk", "an"]:
        f[f"{col}_s1r_a"] = W[f"{col}_s1r"].values[a]
        f[f"{col}_s1r_b"] = W[f"{col}_s1r"].values[b]
        f[f"{col}_poolr_a"] = W[f"{col}_poolr"].values[a]
        f[f"{col}_poolr_b"] = W[f"{col}_poolr"].values[b]
    for ch in COMP_CH:
        s = pairs[f"score_{ch}"].values
        m1, m2 = W[f"{ch}_max1"].values[b], W[f"{ch}_max2"].values[b]
        f[f"{ch}_margin_p"] = np.where(s >= m1, s - m2, s - m1)
        f[f"{ch}_nrel_p"] = W[f"{ch}_nrel"].values[b]
    return pd.DataFrame(f, index=pairs.index).astype(np.float32)


# ----------------------------------------------------------------------------- pruner (cheap features)
def cheap_features(pairs, W, flags):
    """Blocking-stage signals only: retrieval scores/ranks, rarity, pool competition, record flags."""
    df = pd.concat([pairs, world_features(pairs, W)], axis=1)
    b = pairs.p.values
    df["src_b"] = flags["source"][b]
    df["native_b"] = flags["native"][b]
    df["web_b"] = flags["web"][b]
    df["addr_empty_b"] = np.isnan(W["an_s1r"].values[b]).astype(np.float32)
    g = df.groupby("s1")
    for c in ["score_joint", "score_addr", "score_name"]:
        df[f"{c}_gap"] = df[c] - g[c].transform("max")
    df["n_cands"] = g.p.transform("size").astype(np.float32)
    return df


def load_flags(split):
    parts = sorted(work(split, "norm").glob("part-*.parquet"))
    t = pa.concat_tables([pq.read_table(p, columns=["native", "web"]) for p in parts])
    return {"source": load_records(split, ["source"]).source.values.astype(np.float32),
            "native": t.column("native").to_numpy().astype(np.float32),
            "web": t.column("web").to_numpy().astype(np.float32)}


def candidate_chunks(split, s1_ids, Ks=KS, window=250_000):
    """Candidates for the given S1 rows, in windows of contiguous record indices (each window holds
    at most `window` S1 records, so memory is bounded whatever fraction of S1 is requested)."""
    n = int(s1_ids.max()) + 1
    keep = np.zeros(n, bool)
    keep[s1_ids] = True
    rec_s1 = np.flatnonzero(load_records(split, ["source"]).source.values == 1)
    bounds = rec_s1[::window].tolist() + [int(rec_s1[-1]) + 1]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if not keep[lo:min(hi, n)].any():
            continue
        pairs = assemble(split, Ks, 1.0, s1_range=(lo, hi), keep_s1=np.pad(keep, (0, max(0, hi - n))))
        if len(pairs):
            yield pairs.reset_index(drop=True)


def prune(df, bundle):
    prob = predict_chunked(bundle["model"], df, bundle["cols"])
    df = df.assign(pprob=prob.astype(np.float32))
    df["prk"] = df.groupby("s1").pprob.rank(ascending=False, method="first")
    return df[(df.pprob >= bundle["t"]) & (df.prk <= bundle["max_per_s1"])].drop(columns=["prk"])


# ----------------------------------------------------------------------------- matcher features
def full_features(df, strings, source):
    """df: pruned candidates with cheap features. Adds string, number and S1-context (over pruned set)."""
    df = df.reset_index(drop=True)
    base = df[["s1", "p"] + [c for c in df.columns if c.startswith(("score_", "rank_"))]]
    pf = pair_features(base, strings, source)
    ctx = context_features(pd.concat([base[["s1", "p", "score_name", "score_addr"]], pf], axis=1))
    ctx = ctx[[c for c in ctx.columns if c.endswith(("_gap", "_rk")) or c in ("name_addr", "n_cands")]]
    ia, ib = pa.array(df.s1.values), pa.array(df.p.values)
    na, nb = strings.column("num").take(ia).to_pylist(), strings.column("num").take(ib).to_pylist()
    sa = [_street(s) for s in strings.column("an").take(ia).to_pylist()]
    sb = [_street(s) for s in strings.column("an").take(ib).to_pylist()]
    nf = pd.DataFrame(_num_feats(na, nb, sa, sb))
    keep_cheap = df.drop(columns=[c for c in ctx.columns if c in df.columns] + [c for c in pf.columns if c in df.columns])
    return pd.concat([keep_cheap, pf.reset_index(drop=True), ctx.reset_index(drop=True), nf], axis=1)


# ----------------------------------------------------------------------------- run3: shift-robust matcher view
# Rates (count / world size) shift for EVERY record when the world size changes (US test has half the
# S1 of train); plain counts are identical for the typical record. Retrieval ranks and name-channel
# scores depend on how crowded the neighbourhood is (test India/France are more crowded), so the
# matcher does not see them; it compares names directly through the string-similarity features.
SHIFTED = ("rank_", "_s1r_", "_poolr_", "_nrel_p", "score_name", "score_cgram", "score_native")


def matcher_view(df, W):
    b, a = df.p.values, df.s1.values
    out = df[[c for c in df.columns if not any(k in c for k in SHIFTED)]].copy()
    for col in ["nc", "nk", "an"]:
        out[f"{col}_s1n_a"] = W[f"{col}_s1n_raw"].values[a]
        out[f"{col}_s1n_b"] = W[f"{col}_s1n_raw"].values[b]
        out[f"{col}_pooln_a"] = W[f"{col}_pooln_raw"].values[a]
        out[f"{col}_pooln_b"] = W[f"{col}_pooln_raw"].values[b]
    for ch in COMP_CH:
        out[f"{ch}_n_p"] = W[f"{ch}_n_raw"].values[b]
    return out


NON_FEAT = {"s1", "p", "label", "fold", "prob", "pprob"}


def predict_chunked(model, df, cols, chunk=1_000_000):
    out = np.empty(len(df), np.float32)
    for lo in range(0, len(df), chunk):
        out[lo:lo + chunk] = model.predict_proba(df[cols].iloc[lo:lo + chunk].to_numpy(np.float32))[:, 1]
    return out


# ----------------------------------------------------------------------------- steps
def step_pruner():
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    keep_s1 = (rec.source.values == 1) & ~R
    W = world_stats("train", keep_s1, f"sim_{SET}")
    flags = load_flags("train")
    gt = load_gt_pairs()
    gk = np.sort(key(gt.s1, gt.p))
    frames = []
    half = A & (np.random.default_rng(3).random(len(A)) < 0.5)  # deeper lists: half of A keeps memory in check
    A = half
    for pairs in candidate_chunks("train", np.flatnonzero(A)):
        df = cheap_features(pairs, W, flags)
        df["label"] = np.isin(key(df.s1, df.p), gk)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in df.columns if c not in NON_FEAT]
    print(f"pruner set {len(df):,} pairs ({len(df)/A.sum():.1f}/S1), {df.label.mean():.4f} pos, {len(cols)} feats", flush=True)
    fold = (pd.util.hash_array(df.s1.values.astype(np.int64)) % 2).astype(np.int8)
    prob = np.zeros(len(df), np.float32)
    for f in (0, 1):
        m = make_model({"max_iter": 300})
        m.fit(df.loc[fold != f, cols].to_numpy(np.float32), df.label.values[fold != f])
        prob[fold == f] = predict_chunked(m, df.loc[fold == f], cols)
    df["pprob"] = prob
    n_pos = gt[A[gt.s1.values]].shape[0]  # all true pairs of A (incl. those retrieval missed)
    print(f"retrieval recall (deeper lists): {df.label.sum()/n_pos:.4f}")
    df["prk"] = df.groupby("s1").pprob.rank(ascending=False, method="first")
    for t in [0.001, 0.003, 0.01, 0.03]:
        for mx in [8, 10, 12, 15]:
            k = df[(df.pprob >= t) & (df.prk <= mx)]
            print(f"t={t:<6} max={mx:<3} cand/S1={len(k)/A.sum():5.2f} recall={k.label.sum()/n_pos:.4f}", flush=True)
    m = make_model({"max_iter": 300})
    m.fit(df[cols].to_numpy(np.float32), df.label.values)
    joblib.dump({"model": m, "cols": cols, "t": 0.01, "max_per_s1": 12}, work("train", f"pruner_{SET}.joblib"))


def step_trainset():
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    keep_s1 = (rec.source.values == 1) & ~R
    W = world_stats("train", keep_s1, f"sim_{SET}")
    flags = load_flags("train")
    bundle = joblib.load(work("train", f"pruner_{SET}.joblib"))
    strings = load_strings("train")
    source = rec.source.values
    gk = np.sort(key(*load_gt_pairs()[["s1", "p"]].values.T))
    out, t0 = [], time.time()
    for i, pairs in enumerate(candidate_chunks("train", np.flatnonzero(B))):
        df = prune(cheap_features(pairs, W, flags), bundle)
        df = full_features(df, strings, source)
        df["label"] = np.isin(key(df.s1, df.p), gk)
        out.append(df)
        print(f"  B chunk {i+1}: {len(df):,} pruned pairs, {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(work("train", f"trainset_B_{SET}.parquet"), index=False)
    print(f"trainset B: {len(df):,} pairs ({len(df)/B.sum():.2f}/S1), {df.label.mean():.4f} pos")


def evaluate(df, s1_ids, gt, thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)):
    rows = []
    for excl in (False, True):
        d = decide.exclusive(df) if excl else df
        for t in thresholds:
            rows.append({"exclusive": excl, "t": t, **summarize(per_entity_f05(s1_ids, decide.by_threshold(d, t), gt))})
    return pd.DataFrame(rows).sort_values("macro_f05", ascending=False)


def load_trainset(rec, R):
    df = pd.read_parquet(work("train", f"trainset_B_{SET}.parquet"))
    if VIEW:
        df = matcher_view(df, world_stats("train", (rec.source.values == 1) & ~R, f"sim_{SET}"))
    return df


def step_cv(params=None):
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    df = load_trainset(rec, R)
    cols = [c for c in df.columns if c not in NON_FEAT]
    fold = (pd.util.hash_array(df.s1.values.astype(np.int64)) % 2).astype(np.int8)
    prob = np.zeros(len(df), np.float32)
    for f in (0, 1):
        m = make_model({"max_iter": 800, **(params or {})})
        m.fit(df.loc[fold != f, cols].to_numpy(np.float32), df.label.values[fold != f])
        prob[fold == f] = predict_chunked(m, df.loc[fold == f], cols)
        print(f"  fold {f}: {m.n_iter_} iters", flush=True)
    df["prob"] = prob
    gt = load_gt_pairs()
    s1_ids = np.flatnonzero(B)
    gt = gt[B[gt.s1.values]]
    res = evaluate(df, s1_ids, gt)
    print(res.head(10).round(4).to_string(index=False))
    pe = per_entity_f05(s1_ids, decide.by_threshold(decide.exclusive(df), res.iloc[0].t), gt)
    print(pe.assign(c=rec.country.values[pe.index.values]).groupby("c").f05.mean().round(4))
    df[["s1", "p", "label", "prob"]].to_parquet(work("train", f"oof_{RUN}.parquet"), index=False)
    res.to_csv(work("train", f"cv_{RUN}.csv"), index=False)


def step_final(params=None):
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    df = load_trainset(rec, R)
    cols = [c for c in df.columns if c not in NON_FEAT]
    m = make_model({"max_iter": 800, **(params or {})})
    t = time.time()
    m.fit(df[cols].to_numpy(np.float32), df.label.values)
    joblib.dump({"model": m, "cols": cols}, work("train", f"matcher_{RUN}.joblib"))
    print(f"matcher: {m.n_iter_} iters, {time.time()-t:.0f}s")


def step_test(t=0.7, excl=True, tag=RUN):
    rec = load_records("test", ["entity_id", "source", "country"])
    s1_ids = np.flatnonzero(rec.source.values == 1)
    W = world_stats("test", rec.source.values == 1, f"all_{SET}")
    flags = load_flags("test")
    pr = joblib.load(work("train", f"pruner_{SET}.joblib"))
    mt = joblib.load(work("train", f"matcher_{RUN}.joblib"))
    strings = load_strings("test")
    source = rec.source.values
    out, t0 = [], time.time()
    for i, pairs in enumerate(candidate_chunks("test", s1_ids, window=80_000)):
        df = full_features(prune(cheap_features(pairs, W, flags), pr), strings, source)
        if VIEW:
            df = matcher_view(df, W)
        prob = predict_chunked(mt["model"], df, mt["cols"])
        out.append(pd.DataFrame({"s1": df.s1.values, "p": df.p.values, "prob": prob}))
        print(f"  test chunk {i+1}: {len(df):,} pairs, {time.time()-t0:.0f}s", flush=True)
    sc = pd.concat(out, ignore_index=True)
    sc.to_parquet(work("test", f"scores_{tag}.parquet"), index=False)
    write(sc, rec, t, excl, folder=RUN)


def write(sc, rec, t, excl, folder=None):
    ids = rec.entity_id.values
    s1_ids = np.flatnonzero(rec.source.values == 1)
    pred = decide.by_threshold(decide.exclusive(sc) if excl else sc, t)
    out_dir = OUTPUT_DIR / folder if folder else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame, col, fname in ((pred, "matched_entity_ids", "matching_results.tsv"),
                              (sc, "candidate_entity_ids", "candidate_pairs.tsv")):
        f = frame[["s1", "p"]].sort_values(["s1", "p"])
        lists = f.assign(pid=ids[f.p.values]).groupby("s1").pid.agg(",".join).reindex(s1_ids).fillna("")
        with open(out_dir / fname, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for a, b in zip(ids[s1_ids], lists.values):
                fh.write(f"{a}\t{b}\n")
        print(f"wrote {out_dir / fname}: {len(f):,} ids ({len(f)/len(s1_ids):.2f}/S1)")


def step_diag():
    """Old sub1 model (raw counts, trained in the real train world) evaluated on sample A inside the
    simulated test-like world. If this drops to ~0.95, the simulation reproduces the leaderboard gap."""
    from .experiment import features_for
    import json
    cfg = json.load(open(work("train", "..", "..", "business_entity_resolution", "configs", "sub1.json")))
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    base = features_for({**cfg, "feature_version": 2})
    mt = joblib.load(work("train", "model_sub1.joblib"))
    gt = load_gt_pairs()
    s1_ids = np.flatnonzero(A)
    gA = gt[A[gt.s1.values]]
    for tag, keep_s1 in (("real", rec.source.values == 1), ("sim", (rec.source.values == 1) & ~R)):
        W = world_stats("train", keep_s1, tag)
        df = base
        a, b = df.s1.values, df.p.values
        for col in ["nc", "nk", "an"]:
            for side, idx in (("a", a), ("b", b)):
                df[f"{col}_s1n_{side}"] = W[f"{col}_s1n_raw"].values[idx]
                df[f"{col}_pooln_{side}"] = W[f"{col}_pooln_raw"].values[idx]
        for ch in COMP_CH:
            s = df[f"score_{ch}"].values
            m1, m2 = W[f"{ch}_max1"].values[b], W[f"{ch}_max2"].values[b]
            df[f"{ch}_margin_p"] = np.where(s >= m1, s - m2, s - m1)
            df[f"{ch}_n_p"] = W[f"{ch}_n_raw"].values[b]
        df["prob"] = predict_chunked(mt["model"], df, mt["cols"])
        res = evaluate(df[["s1", "p", "prob"]], s1_ids, gA, thresholds=(0.6, 0.7, 0.8, 0.9))
        print(f"[{tag} world] sub1 on sample A (in-sample, so absolute level optimistic):")
        print(res.head(4).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    {"diag": step_diag, "pruner": step_pruner, "trainset": step_trainset, "cv": step_cv, "final": step_final, "test": step_test}[sys.argv[1]]()
