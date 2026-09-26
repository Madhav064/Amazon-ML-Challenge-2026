"""Stage 3: candidate assembly (union of channels) and pairwise features.

String similarities use rapidfuzz.process.cpdist (element-wise, multithreaded C). Everything is computed
in chunks of pairs so memory stays bounded; strings are held as compact Arrow arrays and only
materialized to Python per chunk.
"""
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (registers pa.compute)
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .blocking import cand_path
from .io import load_records
from .paths import work

STR_COLS = ["nn", "nc", "nk", "sk", "an", "num"]


# ----------------------------------------------------------------------------- candidates
def assemble(split: str, Ks: dict, sample: float = 1.0, s1_range=None) -> pd.DataFrame:
    """Union of per-channel top-K lists; one row per (s1, p) with every channel's score and rank."""
    out = None
    keep = None
    if sample < 1 and cand_path(split, next(iter(Ks)), 1.0).exists():
        # full-run candidates restricted to the dev sample: identical to a sampled run (IDF is sample-independent)
        from .blocking import query_mask
        keep = query_mask(split, load_records(split, ["source"]), sample)
    for ch, K in Ks.items():
        flt = [("rank", "<", K)]
        if s1_range is not None:
            flt += [("s1", ">=", int(s1_range[0])), ("s1", "<", int(s1_range[1]))]
        t = pq.read_table(cand_path(split, ch, 1.0 if keep is not None else sample), filters=flt).to_pandas()
        if keep is not None:
            t = t[keep[t.s1.values]]
        t = t.rename(columns={"score": f"score_{ch}", "rank": f"rank_{ch}"})
        out = t if out is None else out.merge(t, on=["s1", "p"], how="outer")
    for ch, K in Ks.items():
        out[f"score_{ch}"] = out[f"score_{ch}"].fillna(0).astype(np.float32)
        out[f"rank_{ch}"] = out[f"rank_{ch}"].fillna(K).astype(np.int16)
    return out.sort_values(["s1", "p"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- features
def load_strings(split: str) -> pa.Table:
    parts = sorted(work(split, "norm").glob("part-*.parquet"))
    return pa.concat_tables([pq.read_table(p, columns=STR_COLS + ["native", "web"]) for p in parts]).combine_chunks()


def _sim(scorer, a, b, scale=100.0):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / np.float32(scale)


def _set_feats(a_list, b_list):
    """Token-set statistics that rapidfuzz does not provide: jaccard, first-token equality, shared count."""
    n = len(a_list)
    jac = np.zeros(n, np.float32); first = np.zeros(n, np.float32); shared = np.zeros(n, np.float32)
    for i in range(n):
        a, b = a_list[i].split(), b_list[i].split()
        if not a or not b:
            jac[i] = first[i] = shared[i] = np.nan
            continue
        sa, sb = set(a), set(b)
        k = len(sa & sb)
        shared[i] = k
        jac[i] = k / len(sa | sb)
        first[i] = a[0] == b[0] or a[0] in sb
    return jac, first, shared


def _alpha(s_list):
    return [" ".join(t for t in s.split() if not t.isdigit()) for s in s_list]


def pair_features(pairs: pd.DataFrame, strings: pa.Table, source: np.ndarray, chunk: int = 1_000_000) -> pd.DataFrame:
    frames = []
    for lo in range(0, len(pairs), chunk):
        pc = pairs.iloc[lo:lo + chunk]
        ia, ib = pa.array(pc.s1.values), pa.array(pc.p.values)
        A = {c: strings.column(c).take(ia).to_pylist() for c in STR_COLS}
        B = {c: strings.column(c).take(ib).to_pylist() for c in STR_COLS}
        f = {}
        f["nn_ratio"] = _sim(fuzz.ratio, A["nn"], B["nn"])
        f["nc_ratio"] = _sim(fuzz.ratio, A["nc"], B["nc"])
        f["nc_tsort"] = _sim(fuzz.token_sort_ratio, A["nc"], B["nc"])
        f["nc_tset"] = _sim(fuzz.token_set_ratio, A["nc"], B["nc"])
        f["nc_partial"] = _sim(fuzz.partial_ratio, A["nc"], B["nc"])
        f["nk_ratio"] = _sim(fuzz.ratio, A["nk"], B["nk"])
        f["nk_jw"] = _sim(JaroWinkler.normalized_similarity, A["nk"], B["nk"], 1.0)
        f["nk_partial"] = _sim(fuzz.partial_ratio, A["nk"], B["nk"])
        f["sk_tsort"] = _sim(fuzz.token_sort_ratio, A["sk"], B["sk"])
        f["sk_tset"] = _sim(fuzz.token_set_ratio, A["sk"], B["sk"])
        f["name_jac"], f["name_first"], f["name_shared"] = _set_feats(A["nc"], B["nc"])
        f["nc_ntok_a"] = np.array([s.count(" ") + 1 for s in A["nc"]], np.float32)
        f["nc_ntok_b"] = np.array([s.count(" ") + 1 for s in B["nc"]], np.float32)

        b_empty = np.array([not s for s in B["an"]])
        f["an_ratio"] = _sim(fuzz.ratio, A["an"], B["an"])
        f["an_tsort"] = _sim(fuzz.token_sort_ratio, A["an"], B["an"])
        f["an_tset"] = _sim(fuzz.token_set_ratio, A["an"], B["an"])
        f["an_partial"] = _sim(fuzz.partial_ratio, A["an"], B["an"])
        aa, ab = _alpha(A["an"]), _alpha(B["an"])
        f["addr_alpha_tset"] = _sim(fuzz.token_set_ratio, aa, ab)
        f["addr_alpha_jac"], _, _ = _set_feats(aa, ab)
        f["num_jac"], f["num_first"], f["num_shared"] = _set_feats(A["num"], B["num"])
        f["num_n_a"] = np.array([len(s.split()) for s in A["num"]], np.float32)
        f["num_n_b"] = np.array([len(s.split()) for s in B["num"]], np.float32)
        f["num_tset"] = _sim(fuzz.token_set_ratio, A["num"], B["num"])
        for k in ["an_ratio", "an_tsort", "an_tset", "an_partial", "addr_alpha_tset", "addr_alpha_jac",
                  "num_jac", "num_first", "num_shared", "num_tset"]:
            f[k][b_empty] = np.nan
        f["addr_empty_b"] = b_empty.astype(np.float32)
        f["src_b"] = source[pc.p.values].astype(np.float32)
        f["native_b"] = strings.column("native").take(ib).to_numpy(zero_copy_only=False).astype(np.float32)
        f["web_b"] = strings.column("web").take(ib).to_numpy(zero_copy_only=False).astype(np.float32)
        frames.append(pd.DataFrame(f, index=pc.index))
    return pd.concat(frames)


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Where does this pair stand among all candidates of the same S1 record?"""
    df["name_addr"] = (df.nc_tset + df.an_tset.fillna(df.an_tset.median())) / 2
    for c in ["name_addr", "nc_tset", "an_tset", "score_name", "score_addr"]:
        g = df.groupby("s1")[c]
        df[f"{c}_gap"] = df[c] - g.transform("max")
        df[f"{c}_rk"] = g.rank(ascending=False, method="min").astype(np.float32)
    df["n_cands"] = df.groupby("s1").p.transform("size").astype(np.float32)
    return df


# ----------------------------------------------------------------------------- v2: rarity + house numbers
def _group_counts(values: pa.ChunkedArray, country: np.ndarray, is_s1: np.ndarray):
    """For every record: how many S1 / pool records of the same country share this exact value."""
    codes = pa.chunked_array(values).dictionary_encode().combine_chunks().indices.to_numpy().astype(np.int64)
    key_ = codes * 8 + country  # value within country
    uniq, inv = np.unique(key_, return_inverse=True)
    s1c = np.bincount(inv, weights=is_s1, minlength=len(uniq))
    poolc = np.bincount(inv, weights=~is_s1, minlength=len(uniq))
    return s1c[inv].astype(np.float32), poolc[inv].astype(np.float32)


def record_rarity(split: str) -> pd.DataFrame:
    path = work(split, "rarity.parquet")
    if path.exists():
        return pd.read_parquet(path)
    rec = load_records(split, ["source", "country"])
    country = rec.country.cat.codes.values.astype(np.int64)
    is_s1 = rec.source.values == 1
    parts = sorted(work(split, "norm").glob("part-*.parquet"))
    out = {}
    for col in ["nc", "nk", "an"]:
        v = pa.concat_tables([pq.read_table(p, columns=[col]) for p in parts]).column(col)
        s1c, pc_ = _group_counts(v, country, is_s1)
        empty = np.asarray(pa.compute.equal(v, "").to_numpy(zero_copy_only=False))
        s1c[empty] = np.nan; pc_[empty] = np.nan
        out[f"{col}_s1n"], out[f"{col}_pooln"] = s1c, pc_
    df = pd.DataFrame(out)
    df.to_parquet(path, index=False)
    return df


def _num_feats(na, nb, sa, sb):
    n = len(na)
    F = {k: np.full(n, np.nan, np.float32) for k in
         ["hn_eq", "hn_lev", "hn_logdiff", "hn_reldiff", "num_min_logdiff", "num_a_in_b", "num_b_in_a", "street_eq", "street_jw"]}
    from rapidfuzz.distance import Levenshtein
    for i in range(n):
        a, b = na[i].split(), nb[i].split()
        if a and b:
            x, y = a[0], b[0]
            F["hn_eq"][i] = x == y
            F["hn_lev"][i] = Levenshtein.distance(x, y)
            xi, yi = int(x[:9]), int(y[:9])
            F["hn_logdiff"][i] = np.log1p(abs(xi - yi))
            F["hn_reldiff"][i] = abs(xi - yi) / max(xi, yi, 1)
            ai, bi = [int(t[:9]) for t in a], [int(t[:9]) for t in b]
            F["num_min_logdiff"][i] = np.log1p(min(abs(u - v) for u in ai for v in bi))
            sa_, sb_ = set(a), set(b)
            F["num_a_in_b"][i] = len(sa_ & sb_) / len(sa_)
            F["num_b_in_a"][i] = len(sa_ & sb_) / len(sb_)
        s, t = sa[i], sb[i]
        if s and t:
            F["street_eq"][i] = s == t
            F["street_jw"][i] = JaroWinkler.normalized_similarity(s, t)
    return F


def _street(an: str) -> str:
    """First alphabetic token right after a number: the street name in '97 liberty st ...'."""
    toks = an.split()
    for i in range(len(toks) - 1):
        if toks[i].isdigit() and toks[i + 1].isalpha() and len(toks[i + 1]) > 2:
            return toks[i + 1]
    return ""


def extra_features_v2(pairs: pd.DataFrame, split: str, strings: pa.Table = None, chunk: int = 1_000_000) -> pd.DataFrame:
    rar = record_rarity(split)
    strings = strings if strings is not None else load_strings(split)
    a, b = pairs.s1.values, pairs.p.values
    f = {}
    for col in ["nc", "nk", "an"]:
        for side, idx in (("a", a), ("b", b)):
            f[f"{col}_s1n_{side}"] = np.log1p(rar[f"{col}_s1n"].values[idx])
            f[f"{col}_pooln_{side}"] = np.log1p(rar[f"{col}_pooln"].values[idx])
    frames = []
    for lo in range(0, len(pairs), chunk):
        ia, ib = pa.array(a[lo:lo + chunk]), pa.array(b[lo:lo + chunk])
        na, nb = strings.column("num").take(ia).to_pylist(), strings.column("num").take(ib).to_pylist()
        sa = [_street(s) for s in strings.column("an").take(ia).to_pylist()]
        sb = [_street(s) for s in strings.column("an").take(ib).to_pylist()]
        frames.append(pd.DataFrame(_num_feats(na, nb, sa, sb)))
    out = pd.concat([pd.DataFrame(f).astype(np.float32), pd.concat(frames, ignore_index=True)], axis=1)
    out.index = pairs.index
    return out


# ----------------------------------------------------------------------------- v3: pool-side competition
V3_CHANNELS = ("joint", "addr", "name")


def pool_competition(split: str, Ks: dict) -> pd.DataFrame:
    """Per pool record, over the FULL candidate lists of all S1 queries (no labels involved):
    best and second-best score it receives per channel, and how many S1 lists contain it."""
    path = work(split, "pool_competition.parquet")
    if path.exists():
        return pd.read_parquet(path)
    n = len(load_records(split, ["source"]))
    out = {}
    for ch in V3_CHANNELS:
        t = pq.read_table(cand_path(split, ch, 1.0), columns=["p", "score", "rank"], filters=[("rank", "<", Ks[ch])])
        p, s = t.column("p").to_numpy(), t.column("score").to_numpy()
        o = np.lexsort((-s, p))  # by pool record, best score first
        p, s = p[o], s[o]
        first = np.r_[True, p[1:] != p[:-1]]
        second = np.r_[False, first[:-1]] & ~first
        m1 = np.zeros(n, np.float32); m2 = np.zeros(n, np.float32)
        m1[p[first]] = s[first]
        m2[p[second]] = s[second]
        out[f"{ch}_max1"], out[f"{ch}_max2"] = m1, m2
        out[f"{ch}_n"] = np.bincount(p, minlength=n).astype(np.float32)
    df = pd.DataFrame(out)
    df.to_parquet(path, index=False)
    return df


def extra_features_v3(pairs: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """margin > 0: this S1 is the pool record's best claimant on that channel; < 0: another S1 beats it."""
    p = pairs.p.values
    f = {}
    for ch in V3_CHANNELS:
        s = pairs[f"score_{ch}"].values
        m1, m2 = comp[f"{ch}_max1"].values[p], comp[f"{ch}_max2"].values[p]
        f[f"{ch}_margin_p"] = np.where(s >= m1, s - m2, s - m1)
        f[f"{ch}_n_p"] = np.log1p(comp[f"{ch}_n"].values[p])
    return pd.DataFrame(f, index=pairs.index).astype(np.float32)


def build(split: str, Ks: dict, sample: float = 1.0, out_name: str = None) -> pd.DataFrame:
    t0 = time.time()
    pairs = assemble(split, Ks, sample)
    t1 = time.time()
    source = load_records(split, ["source"]).source.values
    strings = load_strings(split)
    feats = pair_features(pairs, strings, source)
    df = context_features(pd.concat([pairs, feats], axis=1))
    df = df.astype({c: np.float32 for c in df.columns if df[c].dtype == np.float64})
    if out_name:
        df.to_parquet(work(split, out_name), index=False)
    print(f"[features {split}] {len(df):,} pairs, assemble {t1-t0:.0f}s, features {time.time()-t1:.0f}s", flush=True)
    return df
