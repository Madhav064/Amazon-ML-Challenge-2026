"""Blocking diagnostics on train: pair recall@K per channel and for unions, candidates/S1, oracle macro F0.5.

Pairs are packed into int64 keys (s1 << 24 | p) so 100M-row candidate sets fit in memory.
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .blocking import cand_path, query_mask
from .io import load_gt_pairs, load_records
from .paths import work


def key(s1, p):
    return (np.asarray(s1, np.int64) << 24) | np.asarray(p, np.int64)


def channel_keys(split, ch, K, sample=1.0):
    t = pq.read_table(cand_path(split, ch, sample), columns=["s1", "p", "rank"], filters=[("rank", "<", K)])
    return np.unique(key(t.column("s1").to_numpy(), t.column("p").to_numpy()))


def report(ckeys: np.ndarray, gkeys: np.ndarray, gt: pd.DataFrame, n_true: np.ndarray, n_s1: int, label: str) -> dict:
    hit = np.isin(gkeys, ckeys, assume_unique=True)
    tp = np.bincount(gt.s1.values[hit], minlength=len(n_true))
    s1_mask = n_true >= 0  # n_true is -1 for non-S1 rows
    denom = 0.25 * n_true + tp
    f = np.where(denom > 0, 1.25 * tp / np.where(denom > 0, denom, 1), 1.0)[s1_mask]
    r = {"blocking": label, "pair_recall": hit.mean(), "cands_per_s1": len(ckeys) / n_s1,
         "entities_full_recall": float((tp[s1_mask] == n_true[s1_mask]).mean()), "oracle_macro_f05": float(f.mean())}
    print(f"{label:34s} recall={r['pair_recall']:.4f}  cand/S1={r['cands_per_s1']:6.1f}  "
          f"entity_full_recall={r['entities_full_recall']:.4f}  oracle_F05={r['oracle_macro_f05']:.4f}", flush=True)
    return r


def main(channels, Ks, sample=1.0, split="train"):
    rec = load_records(split, ["source", "country"])
    s1 = query_mask(split, rec, sample)
    gt = load_gt_pairs()
    gt = gt[s1[gt.s1.values]].reset_index(drop=True)
    gkeys = key(gt.s1, gt.p)
    n_true = np.full(len(rec), -1, np.int64)
    n_true[s1] = 0
    n_true += np.bincount(gt.s1.values, minlength=len(rec)) * s1
    rows = []
    for K in Ks:
        ks = {ch: channel_keys(split, ch, K, sample) for ch in channels}
        for ch in channels:
            rows.append(report(ks[ch], gkeys, gt, n_true, s1.sum(), f"{ch}@{K}"))
        if len(channels) > 1:
            u = np.unique(np.concatenate(list(ks.values())))
            rows.append(report(u, gkeys, gt, n_true, s1.sum(), "+".join(channels) + f"@{K}"))
    u = np.unique(np.concatenate([channel_keys(split, ch, max(Ks), sample) for ch in channels]))
    hit = np.isin(gkeys, u)
    by = pd.DataFrame({"country": rec.country.values[gt.s1.values], "src": rec.source.values[gt.p.values], "hit": hit})
    print("recall of union@max(K) by country/source:\n", by.groupby(["country", "src"]).hit.mean().round(4).to_string())
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("channels", nargs="+")
    ap.add_argument("--Ks", default="5,10,20,50")
    ap.add_argument("--sample", type=float, default=1.0)
    a = ap.parse_args()
    df = main(a.channels, [int(k) for k in a.Ks.split(",")], a.sample)
    df.to_csv(work("train", f"blocking_report_{'_'.join(a.channels)}_s{a.sample}.csv"), index=False)
