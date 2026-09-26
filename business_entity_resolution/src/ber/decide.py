"""Stage 5: turn pair probabilities into per-S1 match lists, optimised for macro F0.5.

Rules (all optional, compared on validation):
  * exclusive  - every S2/S3 record belongs to at most one S1 entity in the ground truth, so a pool
                 record is only kept for the S1 that gives it the highest probability.
  * threshold  - global probability cut-off.
  * expected-F - per S1, choose the top-k (by probability) that maximises the expected F0.5
                 1.25*sum(p_1..p_k) / (0.25*E[n_true] + k), with k=0 scoring P(no true match).
"""
import numpy as np
import pandas as pd


def exclusive(df: pd.DataFrame, prob="prob") -> pd.DataFrame:
    best = df.groupby("p")[prob].transform("max")
    return df[df[prob] >= best].drop_duplicates("p")


def by_threshold(df: pd.DataFrame, t: float, prob="prob") -> pd.DataFrame:
    return df[df[prob] >= t][["s1", "p"]]


def by_expected_f(df: pd.DataFrame, prob="prob", floor: float = 0.0) -> pd.DataFrame:
    d = df[["s1", "p", prob]].sort_values(["s1", prob], ascending=[True, False])
    p = d[prob].values.astype(np.float64)
    s1 = d.s1.values
    start = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1]
    keep = np.zeros(len(d), bool)
    for a, b in zip(start, np.r_[start[1:], len(d)]):
        ps = p[a:b]
        exp_true = ps.sum()
        best_k, best = 0, np.prod(1 - ps)  # k = 0: right only when the entity is a singleton
        cs = np.cumsum(ps)
        ks = np.arange(1, len(ps) + 1)
        ef = 1.25 * cs / (0.25 * exp_true + ks)
        k = int(np.argmax(ef))
        if ef[k] > best:
            best_k = k + 1
        keep[a:a + best_k] = True
    out = d[keep]
    return out[out[prob] >= floor][["s1", "p"]]
