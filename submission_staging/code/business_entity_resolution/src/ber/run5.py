"""Final matcher: trained on ~1.7x more entities (sample B + sample C), then scores the test set.
Artifacts are named after cascade.SET (final submission: run6 = 7 forward channels + reverse retrieval).

    python -m ber.run5 trainset   # pruned + full features for sample C (disjoint from A, B, R)
    python -m ber.run5 final      # matcher_<SET> on B + C (shift-robust feature view)
    python -m ber.run5 test       # scores_<SET> for every pruned test candidate
    python -m ber.finalize run6
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .blocking_eval import key
from .cascade import (NON_FEAT, SET, candidate_chunks, cheap_features, full_features, load_flags, load_trainset, masks,
                      matcher_view, predict_chunked, prune, world_stats)
from .experiment import make_model
from .features import load_strings
from .io import load_gt_pairs, load_records
from .paths import work

TAG = SET  # run5 = candidate set run2; run6 = run2 channels + reverse retrieval


def sample_c(rec):
    A, B, R = masks(rec)
    rest = (rec.source.values == 1) & ~A & ~B & ~R
    return rest & (np.random.default_rng(4).random(len(rec)) < 0.5)


def step_trainset():
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    C = sample_c(rec)
    W = world_stats("train", (rec.source.values == 1) & ~R, f"sim_{SET}")
    flags = load_flags("train")
    bundle = joblib.load(work("train", f"pruner_{SET}.joblib"))
    strings = load_strings("train")
    gk = np.sort(key(*load_gt_pairs()[["s1", "p"]].values.T))
    out, t0 = [], time.time()
    for i, pairs in enumerate(candidate_chunks("train", np.flatnonzero(C))):
        df = full_features(prune(cheap_features(pairs, W, flags), bundle), strings, rec.source.values)
        df["label"] = np.isin(key(df.s1, df.p), gk)
        out.append(df)
        print(f"  C chunk {i+1}: {len(df):,} pairs, {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(work("train", f"trainset_C_{SET}.parquet"), index=False)
    print(f"trainset C: {len(df):,} pairs from {C.sum():,} S1")


def step_final():
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    W = world_stats("train", (rec.source.values == 1) & ~R, f"sim_{SET}")
    b = load_trainset(rec, R)
    # score_joint_rev is left out: its value is already copied into score_joint (same cosine)
    cols = [c for c in b.columns if c not in NON_FEAT and c != "score_joint_rev"]
    nb = len(b)
    nc = pq.ParquetFile(work("train", f"trainset_C_{SET}.parquet")).metadata.num_rows
    X = np.empty((nb + nc, len(cols)), np.float32)  # filled in place: no second copy of the data
    y = np.empty(nb + nc, bool)
    for j, c in enumerate(cols):
        X[:nb, j] = b[c].values
    y[:nb] = b.label.values
    del b
    c = matcher_view(pd.read_parquet(work("train", f"trainset_C_{SET}.parquet")), W)
    for j, col in enumerate(cols):
        X[nb:, j] = c[col].values
    y[nb:] = c.label.values
    del c
    t = time.time()
    # all earlier runs used the full 800 iterations; no early-stopping split (it would copy X)
    m = make_model({"max_iter": 800, "early_stopping": False})
    m.fit(X, y)
    joblib.dump({"model": m, "cols": cols}, work("train", f"matcher_{TAG}.joblib"))
    print(f"matcher_{TAG} on {len(y):,} pairs: {m.n_iter_} iters, {time.time()-t:.0f}s")


def step_test():
    rec = load_records("test", ["source"])
    s1_ids = np.flatnonzero(rec.source.values == 1)
    W = world_stats("test", rec.source.values == 1, f"all_{SET}")
    flags = load_flags("test")
    pr = joblib.load(work("train", f"pruner_{SET}.joblib"))
    mt = joblib.load(work("train", f"matcher_{TAG}.joblib"))
    strings = load_strings("test")
    out, t0 = [], time.time()
    for i, pairs in enumerate(candidate_chunks("test", s1_ids, window=80_000)):
        df = matcher_view(full_features(prune(cheap_features(pairs, W, flags), pr), strings, rec.source.values), W)
        out.append(pd.DataFrame({"s1": df.s1.values, "p": df.p.values, "prob": predict_chunked(mt["model"], df, mt["cols"])}))
        print(f"  test chunk {i+1}: {len(df):,} pairs, {time.time()-t0:.0f}s", flush=True)
    pd.concat(out, ignore_index=True).to_parquet(work("test", f"scores_{TAG}.parquet"), index=False)
    print(f"wrote scores_{TAG}")


if __name__ == "__main__":
    {"trainset": step_trainset, "final": step_final, "test": step_test}[sys.argv[1]]()
