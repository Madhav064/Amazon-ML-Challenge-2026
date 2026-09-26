"""Stage 2: re-score each pair in the context of its S1 entity's other candidates.

Stage 1 decides pair by pair, but the metric is per entity. Stage 2 sees, for every pair, the
stage-1 probabilities of its siblings (same S1): how many confident matches the entity already has,
overall and from the same source, and how this pair ranks among them. Trained on out-of-fold stage-1
probabilities with the same S1-grouped folds, so its inputs look like unseen-entity predictions.
"""
import numpy as np
import pandas as pd

KEEP_RAW = ["nc_tset", "an_tset", "hn_logdiff", "name_jac", "addr_empty_b", "src_b", "nc_s1n_b", "an_s1n_b",
            "joint_margin_p", "addr_margin_p", "name_margin_p", "score_joint"]


def _second_values(s1: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Second-highest probability within each row's S1 group (0 when the group has one candidate)."""
    o = np.lexsort((-p, s1))
    s1s, ps = s1[o], p[o]
    first = np.r_[True, s1s[1:] != s1s[:-1]]
    start = np.flatnonzero(first)
    size = np.diff(np.r_[start, len(s1s)])
    sec = np.where(size > 1, ps[np.minimum(start + 1, len(ps) - 1)], 0.0)
    out = np.empty(len(p), np.float32)
    out[o] = sec[np.cumsum(first) - 1]
    return out


def context(df: pd.DataFrame, prob="prob") -> pd.DataFrame:
    d = df[["s1", "src_b", prob]].copy()
    d["hi"] = (d[prob] > 0.5).astype(np.float32)
    g = d.groupby("s1")[prob]
    mx = g.transform("max").values
    f = pd.DataFrame(index=df.index)
    f["st_prob"] = d[prob].values
    f["st_sum_other"] = g.transform("sum").values - d[prob].values
    f["st_max_other"] = np.where(d[prob].values >= mx, _second_values(d.s1.values, d[prob].values), mx)
    f["st_rank"] = g.rank(ascending=False, method="first").values
    f["st_nhi_other"] = d.groupby("s1").hi.transform("sum").values - d.hi.values
    gs = d.groupby(["s1", "src_b"])
    f["st_src_sum_other"] = gs[prob].transform("sum").values - d[prob].values
    f["st_src_nhi_other"] = gs.hi.transform("sum").values - d.hi.values
    f["st_src_rank"] = gs[prob].rank(ascending=False, method="first").values
    for c in KEEP_RAW:
        if c in df:
            f[c] = df[c].values
    return f.astype(np.float32)
