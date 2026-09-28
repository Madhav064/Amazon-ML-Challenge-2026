"""False-positive / false-negative analysis of an experiment's OOF predictions.

    python -m ber.analysis <experiment_name> [--t 0.5] [--exclusive] [--n 15]

FP types:  into_singleton  - S1 has no true match at all (costs the entity its full 1.0)
           owned_elsewhere - pool record truly belongs to a different S1
           distractor      - pool record matches no S1 at all
FN types:  blocking_miss   - true pair never reached the candidate set
           below_threshold - candidate scored under the decision rule
           lost_exclusive  - dropped because another S1 claimed the pool record more strongly
"""
import argparse

import numpy as np
import pandas as pd

from . import decide
from .blocking import query_mask
from .blocking_eval import key
from .io import load_gt_pairs, load_records
from .metrics import per_entity_f05, summarize
from .paths import work


def analyse(name, t=0.5, excl=True, n=15, rule="threshold"):
    oof = pd.read_parquet(work("train", f"oof_{name}.parquet"))
    rec = load_records("train", ["source", "country", "name", "address"])
    s1_ids = np.flatnonzero(query_mask("train", rec, 0.1))
    gt_all = load_gt_pairs()
    owner = pd.Series(gt_all.s1.values, index=gt_all.p.values)
    gt = gt_all[np.isin(gt_all.s1.values, s1_ids)]

    d = decide.exclusive(oof) if excl else oof
    pred = decide.by_threshold(d, t) if rule == "threshold" else decide.by_expected_f(d)
    pe = per_entity_f05(s1_ids, pred, gt)
    print("overall:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summarize(pe).items()})
    pe["country"] = rec.country.values[pe.index.values]
    print("\nby country:\n", pe.groupby("country").agg(f05=("f05", "mean"), n=("f05", "size"),
                                                      singleton_rate=("n_true", lambda x: (x == 0).mean())).round(4))
    pe["bucket"] = pd.cut(pe.n_true, [-1, 0, 1, 2, 3, 4, 20], labels=["0", "1", "2", "3", "4", "5+"])
    print("\nby number of true matches:\n", pe.groupby("bucket").agg(f05=("f05", "mean"), n=("f05", "size"),
                                                                  mean_pred=("n_pred", "mean")).round(3))

    pk, gk = key(pred.s1, pred.p), key(gt.s1, gt.p)
    fp = pred[~np.isin(pk, gk)].copy()
    n_true = pe.n_true
    fp["type"] = np.where(n_true.reindex(fp.s1).values == 0, "into_singleton",
                          np.where(owner.reindex(fp.p).notna().values, "owned_elsewhere", "distractor"))
    fn = gt[~np.isin(gk, pk)].copy()
    ok = key(oof.s1, oof.p)
    in_c = np.isin(key(fn.s1, fn.p), ok)
    in_d = np.isin(key(fn.s1, fn.p), key(d.s1, d.p))
    fn["type"] = np.where(~in_c, "blocking_miss", np.where(~in_d, "lost_exclusive", "below_threshold"))
    print(f"\nFP pairs {len(fp):,}:\n", fp.type.value_counts().to_string())
    print(f"FN pairs {len(fn):,}:\n", fn.type.value_counts().to_string())

    p_of = oof.set_index(["s1", "p"]).prob
    def show(rows, title):
        print(f"\n--- {title}")
        for _, r in rows.iterrows():
            pr = p_of.get((r.s1, r.p), np.nan)
            print(f"[{rec.country[r.s1]}] p={pr:.3f}  A: {rec['name'][r.s1]} | {rec.address[r.s1]}\n"
                  f"{'':14}B: {rec['name'][r.p]} | {rec.address[r.p]}")
    for ty in ["into_singleton", "owned_elsewhere", "distractor"]:
        show(fp[fp.type == ty].sample(min(n, (fp.type == ty).sum()), random_state=0), f"FP {ty}")
    for ty in ["below_threshold", "lost_exclusive", "blocking_miss"]:
        show(fn[fn.type == ty].sample(min(n, (fn.type == ty).sum()), random_state=0), f"FN {ty}")
    return pe, fp, fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--rule", default="threshold")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    analyse(a.name, a.t, a.exclusive, a.n, a.rule)
