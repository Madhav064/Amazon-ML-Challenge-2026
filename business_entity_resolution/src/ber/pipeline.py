"""End-to-end pipeline: blocking -> features -> scoring -> decisions -> output files.

    python -m ber.pipeline block    <split>              # all retrieval channels, all S1 queries
    python -m ber.pipeline train    <config.json>        # fit final model on the dev-sample features
    python -m ber.pipeline score    <config.json> <split> # features + probabilities for every candidate pair
    python -m ber.pipeline holdout  <config.json>        # full-competition evaluation on train S1 unseen in training
    python -m ber.pipeline write    <config.json>        # decisions on test scores -> output/*.tsv

Scoring runs in S1-index chunks so features for ~60M pairs never sit in memory at once; the
S1-context features are per-S1 groups, so chunking on S1 boundaries leaves them exact.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import decide, stack
from .blocking import CHANNELS, query_mask, run_channel
from .experiment import feature_list, features_for, label, make_model
from .features import (assemble, context_features, extra_features_v2, extra_features_v3, load_strings,
                       pair_features, pool_competition)
from .io import load_gt_pairs, load_records
from .metrics import per_entity_f05, summarize
from .paths import OUTPUT_DIR, work


def load_cfg(path):
    return json.loads(Path(path).read_text())


def block(split, Ks):
    # train keeps deeper lists so larger-K variants can be evaluated without re-running retrieval
    for ch, K in Ks.items():
        run_channel(split, ch, K=max(K, 30 if ch == "joint" else 20) if split == "train" else K, sample=1.0)


def train_final(cfg):
    df = features_for(cfg)
    rec = load_records("train", ["source"])
    s1_ids = np.flatnonzero(query_mask("train", rec, cfg["sample"]))
    gt = load_gt_pairs()
    df["label"] = label(df, gt[np.isin(gt.s1.values, s1_ids)])
    cols = feature_list(df, cfg)
    m = make_model({**(cfg.get("model") or {}), **(cfg.get("final_model") or {})})
    t = time.time()
    m.fit(df[cols].values, df.label.values)
    bundle = {"model": m, "cols": cols, "cfg": cfg}
    print(f"trained stage-1 on {len(df):,} pairs ({m.n_iter_} iters) in {time.time()-t:.0f}s", flush=True)
    if cfg.get("stack"):
        # stage 2 learns from OUT-OF-FOLD stage-1 probabilities (saved by the CV experiment of the same config)
        oof = pd.read_parquet(work("train", f"oof_{cfg['oof_from']}.parquet"), columns=["s1", "p", "prob1"])
        df = df.merge(oof.rename(columns={"prob1": "prob"}), on=["s1", "p"], how="left")
        assert df.prob.notna().all()
        X2 = stack.context(df)
        m2 = make_model({"max_iter": 300, "max_leaf_nodes": 31, **(cfg.get("stack_model") or {})})
        m2.fit(X2.values, df.label.values)
        bundle.update(model2=m2, cols2=list(X2.columns))
        print(f"trained stage-2 ({m2.n_iter_} iters)", flush=True)
    joblib.dump(bundle, work("train", f"model_{cfg['name']}.joblib"))


def score(cfg, split, exclude_sample=None, chunk=120_000):
    bundle = joblib.load(work("train", f"model_{cfg['name']}.joblib"))
    model, cols = bundle["model"], bundle["cols"]
    rec = load_records(split, ["source"])
    s1 = np.flatnonzero(rec.source.values == 1)
    if exclude_sample is not None:  # train holdout: skip S1 the model was trained on
        s1 = s1[~query_mask(split, rec, exclude_sample)[s1]]
    source = rec.source.values
    del rec
    strings = load_strings(split)
    comp = pool_competition(split, cfg["blocking"]) if cfg.get("feature_version", 1) >= 3 else None
    out, t0 = [], time.time()
    bounds = s1[::chunk].tolist() + [s1[-1] + 1]
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        pairs = assemble(split, cfg["blocking"], 1.0, s1_range=(lo, hi))
        if exclude_sample is not None:
            pairs = pairs[np.isin(pairs.s1.values, s1)].reset_index(drop=True)
        if len(pairs) == 0:
            continue
        df = context_features(pd.concat([pairs, pair_features(pairs, strings, source)], axis=1))
        if cfg.get("feature_version", 1) >= 2:
            df = pd.concat([df, extra_features_v2(df[["s1", "p"]], split, strings)], axis=1)
        if cfg.get("feature_version", 1) >= 3:
            df = pd.concat([df, extra_features_v3(df, comp)], axis=1)
        prob = model.predict_proba(df[cols].values.astype(np.float32))[:, 1].astype(np.float32)
        res = {"s1": df.s1.values, "p": df.p.values, "prob": prob}
        if "model2" in bundle:
            df["prob"] = prob
            X2 = stack.context(df)[bundle["cols2"]]
            res["prob1"] = prob
            res["prob"] = bundle["model2"].predict_proba(X2.values)[:, 1].astype(np.float32)
        out.append(pd.DataFrame(res))
        print(f"  chunk {i+1}/{len(bounds)-1}: {len(df):,} pairs, {time.time()-t0:.0f}s", flush=True)
    res = pd.concat(out, ignore_index=True)
    tag = "holdout" if exclude_sample is not None else split
    res.to_parquet(work(split, f"scores_{cfg['name']}_{tag}.parquet"), index=False)
    return res


def holdout(cfg):
    sc = pd.read_parquet(work("train", f"scores_{cfg['name']}_holdout.parquet"))
    rec = load_records("train", ["source", "country"])
    s1 = rec.source.values == 1
    s1_ids = np.flatnonzero(s1 & ~query_mask("train", rec, cfg["sample"]))
    gt = load_gt_pairs()
    gt = gt[np.isin(gt.s1.values, s1_ids)]
    rows = []
    for excl in (False, True):
        d = decide.exclusive(sc) if excl else sc
        for t in cfg.get("thresholds", [round(x, 2) for x in np.arange(0.3, 0.91, 0.05)]):
            pe = per_entity_f05(s1_ids, decide.by_threshold(d, t), gt)
            r = {"exclusive": excl, "t": t, **summarize(pe)}
            for c in ["India", "US"]:
                r[f"f05_{c}"] = float(pe.f05[rec.country.values[pe.index.values] == c].mean())
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(work("train", f"holdout_{cfg['name']}.csv"), index=False)
    best = res.sort_values("macro_f05", ascending=False).iloc[0]
    print(f"HOLDOUT best macro F0.5 {best.macro_f05:.4f} at t={best.t} exclusive={best.exclusive}")
    return res


def write_outputs(cfg, t=None, excl=None):
    t = cfg["decision"]["threshold"] if t is None else t
    excl = cfg["decision"]["exclusive"] if excl is None else excl
    rec = load_records("test", ["entity_id", "source"])
    ids = rec.entity_id.values
    s1_ids = np.flatnonzero(rec.source.values == 1)
    sc = pd.read_parquet(work("test", f"scores_{cfg['name']}_test.parquet"))
    d = decide.exclusive(sc) if excl else sc
    pred = decide.by_threshold(d, t)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame, col, fname in ((pred, "matched_entity_ids", "matching_results.tsv"),
                              (sc, "candidate_entity_ids", "candidate_pairs.tsv")):
        f = frame[["s1", "p"]].sort_values(["s1", "p"])
        lists = f.assign(pid=ids[f.p.values]).groupby("s1").pid.agg(",".join)
        full = pd.Series("", index=s1_ids).add(lists.reindex(s1_ids).fillna(""), fill_value="")
        out = pd.DataFrame({"source1_entity_id": ids[s1_ids], col: full.values})
        with open(OUTPUT_DIR / fname, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for a, b in zip(out.source1_entity_id.values, out[col].values):
                fh.write(f"{a}\t{b}\n")
        print(f"wrote {OUTPUT_DIR / fname}: {len(out):,} rows, {len(f):,} ids")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "block":
        cfg = load_cfg(sys.argv[3]) if len(sys.argv) > 3 else None
        block(sys.argv[2], cfg["blocking"] if cfg else {"joint": 20, "addr": 10, "name": 5, "cgram": 5, "name_noaddr": 5})
    elif cmd == "train":
        train_final(load_cfg(sys.argv[2]))
    elif cmd == "score":
        cfg = load_cfg(sys.argv[2])
        split = sys.argv[3]
        score(cfg, "train", exclude_sample=cfg["sample"]) if split == "holdout" else score(cfg, split)
    elif cmd == "holdout":
        holdout(load_cfg(sys.argv[2]))
    elif cmd == "write":
        write_outputs(load_cfg(sys.argv[2]))
