"""Reproducible experiment runner.

    python -m ber.experiment configs/<name>.json

A config fixes: blocking (per-channel K), dev sample of S1 queries, feature subset, model params,
number of folds, decision rules. Results (config + metrics) are appended to experiments/results.jsonl
and OOF predictions saved for error analysis. Folds are grouped by S1 entity, so no S1 contributes
pairs to both the training and the evaluation side of a fold.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import decide, stack
from .blocking import query_mask
from .blocking_eval import key
from .features import build, extra_features_v2, extra_features_v3, pool_competition
from .io import load_gt_pairs, load_records
from .metrics import per_entity_f05, summarize
from .paths import ROOT, work

EXP_DIR = ROOT / "experiments"
NON_FEATURES = {"s1", "p", "label", "fold", "prob", "prob1"}


def features_for(cfg) -> pd.DataFrame:
    """v1 features are cached per (blocking, sample); each later feature block is cached separately
    and joined row-aligned, so adding a block never recomputes the earlier ones."""
    # "cand2": candidates from sample-independent IDF (full-run lists filtered to the dev sample)
    tag = hashlib.md5(json.dumps([cfg["blocking"], cfg["sample"], 1, "cand2"], sort_keys=True).encode()).hexdigest()[:8]
    path = work("train", f"feat_{tag}.parquet")
    df = pd.read_parquet(path) if path.exists() else build("train", cfg["blocking"], cfg["sample"], out_name=path.name)
    if cfg.get("feature_version", 1) >= 2:
        p2 = work("train", f"featx2_{tag}.parquet")
        if not p2.exists():
            t = time.time()
            extra_features_v2(df[["s1", "p"]], "train").to_parquet(p2, index=False)
            print(f"  v2 features in {time.time() - t:.0f}s", flush=True)
        df = pd.concat([df, pd.read_parquet(p2)], axis=1)
    if cfg.get("feature_version", 1) >= 3:
        df = pd.concat([df, extra_features_v3(df, pool_competition("train", cfg["blocking"]))], axis=1)
    return df


def label(df: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    return np.isin(key(df.s1, df.p), key(gt.s1, gt.p))


def feature_list(df, cfg):
    cols = [c for c in df.columns if c not in NON_FEATURES]
    if cfg.get("features"):
        cols = [c for c in cols if c in set(cfg["features"])]
    return [c for c in cols if c not in set(cfg.get("drop_features", []))]


def make_model(params):
    p = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100, l2_regularization=1.0,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=0)
    p.update(params or {})
    return HistGradientBoostingClassifier(**p)


def oof_predict(df, cols, cfg):
    k = cfg.get("folds", 4)
    fold = (pd.util.hash_array(df.s1.values.astype(np.int64)) % k).astype(np.int8)  # grouped by S1
    prob = np.zeros(len(df), np.float32)
    models = []
    for f in range(k):
        tr, va = fold != f, fold == f
        m = make_model(cfg.get("model"))
        m.fit(df.loc[tr, cols].values, df.label.values[tr])
        prob[va] = m.predict_proba(df.loc[va, cols].values)[:, 1]
        models.append(m)
        print(f"  fold {f}: {m.n_iter_} iters", flush=True)
    return prob, fold, models


def evaluate_decisions(df, s1_ids, gt, cfg):
    rows = []
    thresholds = cfg.get("thresholds", [round(x, 2) for x in np.arange(0.2, 0.96, 0.05)])
    for excl in (False, True):
        d = decide.exclusive(df) if excl else df
        for t in thresholds:
            s = summarize(per_entity_f05(s1_ids, decide.by_threshold(d, t), gt))
            rows.append({"rule": "threshold", "exclusive": excl, "t": t, **s})
        s = summarize(per_entity_f05(s1_ids, decide.by_expected_f(d), gt))
        rows.append({"rule": "expected_f", "exclusive": excl, "t": None, **s})
    return pd.DataFrame(rows)


def run(cfg_path: str):
    cfg = json.loads(Path(cfg_path).read_text())
    t0 = time.time()
    rec = load_records("train", ["source"])
    s1_ids = np.flatnonzero(query_mask("train", rec, cfg["sample"]))
    gt = load_gt_pairs()
    gt = gt[np.isin(gt.s1.values, s1_ids)]
    df = features_for(cfg)
    df["label"] = label(df, gt)
    cols = feature_list(df, cfg)
    print(f"[{cfg['name']}] {len(df):,} pairs, {df.label.mean():.4f} positive, {len(cols)} features", flush=True)
    if cfg.get("baseline_score"):  # non-ML baseline: rank/threshold a single similarity column
        df["prob"], df["fold"], models = df[cfg["baseline_score"]].astype(np.float32), 0, []
    else:
        df["prob"], df["fold"], models = oof_predict(df, cols, cfg)
    stage1 = None
    if cfg.get("stack"):
        r1 = evaluate_decisions(df, s1_ids, gt, {**cfg, "thresholds": [0.6, 0.65, 0.7, 0.75]})
        stage1 = r1.sort_values("macro_f05", ascending=False).iloc[0].to_dict()
        print(f"  stage-1 best {stage1['macro_f05']:.4f} (t={stage1['t']}, exclusive={stage1['exclusive']})", flush=True)
        X2 = stack.context(df)
        df["prob1"] = df.prob
        prob2 = np.zeros(len(df), np.float32)
        for f in range(cfg.get("folds", 4)):
            tr, va = df.fold.values != f, df.fold.values == f
            m2 = make_model({"max_iter": 300, "max_leaf_nodes": 31, **(cfg.get("stack_model") or {})})
            m2.fit(X2.values[tr], df.label.values[tr])
            prob2[va] = m2.predict_proba(X2.values[va])[:, 1]
        df["prob"] = prob2

    ceiling = summarize(per_entity_f05(s1_ids, df[df.label][["s1", "p"]], gt))["macro_f05"]
    res = evaluate_decisions(df, s1_ids, gt, cfg)
    best = res.sort_values("macro_f05", ascending=False).iloc[0].to_dict()
    imp = (feature_importance(models[0], df[df.fold == 0].sample(min(200_000, (df.fold == 0).sum()), random_state=0), cols)
           if models else pd.Series(dtype=float))

    EXP_DIR.mkdir(exist_ok=True)
    out = {"name": cfg["name"], "time_s": round(time.time() - t0), "n_pairs": len(df), "blocking_ceiling_f05": ceiling,
           "best": best, "stage1_best": stage1, "config": cfg}
    with open(EXP_DIR / "results.jsonl", "a") as fh:
        fh.write(json.dumps(out, default=str) + "\n")
    res.to_csv(EXP_DIR / f"{cfg['name']}_decisions.csv", index=False)
    imp.to_csv(EXP_DIR / f"{cfg['name']}_importance.csv")
    keep = ["s1", "p", "label", "prob", "fold"] + (["prob1"] if "prob1" in df else [])
    df[keep].to_parquet(work("train", f"oof_{cfg['name']}.parquet"), index=False)
    print(res.sort_values("macro_f05", ascending=False).head(8).to_string(index=False))
    print(f"[{cfg['name']}] blocking ceiling {ceiling:.4f}  best macro F0.5 {best['macro_f05']:.4f} "
          f"({best['rule']}, exclusive={best['exclusive']}, t={best['t']})  {time.time()-t0:.0f}s")
    return out


def feature_importance(model, sample, cols, repeats=2):
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, sample[cols].values, sample.label.values, scoring="average_precision",
                               n_repeats=repeats, random_state=0, n_jobs=1)
    return pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)


if __name__ == "__main__":
    run(sys.argv[1])
