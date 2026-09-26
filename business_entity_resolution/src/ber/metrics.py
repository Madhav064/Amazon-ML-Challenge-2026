"""Exact competition metric: per-Source-1-entity F0.5, macro-averaged over ALL evaluated S1 entities.

Per entity: F0.5 = 1.25*P*R / (0.25*P + R) = 1.25*tp / (0.25*n_true + n_pred);
empty prediction on an entity with no true matches scores 1.0.
"""
import numpy as np
import pandas as pd


def per_entity_f05(s1_ids: np.ndarray, pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """s1_ids: all evaluated S1 idx. pred/truth: DataFrames with columns s1, p (unique pairs)."""
    s1_ids = np.asarray(s1_ids)
    pred = pred[["s1", "p"]].drop_duplicates()
    truth = truth[truth.s1.isin(s1_ids)][["s1", "p"]]
    pred = pred[pred.s1.isin(s1_ids)]
    tp = pred.merge(truth, on=["s1", "p"]).groupby("s1").size()
    out = pd.DataFrame(index=pd.Index(s1_ids, name="s1"))
    out["n_pred"] = pred.groupby("s1").size().reindex(out.index, fill_value=0)
    out["n_true"] = truth.groupby("s1").size().reindex(out.index, fill_value=0)
    out["tp"] = tp.reindex(out.index, fill_value=0)
    denom = 0.25 * out.n_true + out.n_pred
    f = np.where(denom > 0, 1.25 * out.tp / denom.where(denom > 0, 1), 1.0)
    out["f05"] = f
    return out


def summarize(pe: pd.DataFrame) -> dict:
    single = pe.n_true == 0
    tp, npred, ntrue = pe.tp.sum(), pe.n_pred.sum(), pe.n_true.sum()
    return {
        "macro_f05": float(pe.f05.mean()),
        "f05_matched_entities": float(pe.f05[~single].mean()) if (~single).any() else float("nan"),
        "f05_singletons": float(pe.f05[single].mean()) if single.any() else float("nan"),
        "pair_precision": float(tp / npred) if npred else float("nan"),
        "pair_recall": float(tp / ntrue) if ntrue else float("nan"),
        "n_entities": int(len(pe)),
        "n_pred_pairs": int(npred),
    }
