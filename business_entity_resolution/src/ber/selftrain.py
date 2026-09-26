"""France self-training (run4). No labels are used: France exists only in the unlabeled test set.

1. features  : matcher features for every pruned France test pair, scored by the run3 matcher.
2. pseudo    : confident positives   = prob >= 0.995 and no other S1 claims the record more strongly;
               confident negatives   = prob <= 0.05, or the record is claimed more strongly (prob >= 0.9)
                                       by another S1 (look-alike decoys);
               everything in between is left out.
3. train     : run3 train set (US/India, true labels) + France pseudo-labelled pairs -> matcher_run4.
4. score     : re-score France pairs with matcher_run4; US/India keep run3 probabilities.

    python -m ber.selftrain features | train | score
"""
import sys
import time

import joblib
import numpy as np
import pandas as pd

from .cascade import (candidate_chunks, cheap_features, full_features, load_flags, load_trainset, masks,
                      matcher_view, predict_chunked, prune, world_stats, NON_FEAT, SET)
from .experiment import make_model
from .features import load_strings
from .io import load_records
from .paths import work


def step_features():
    rec = load_records("test", ["source", "country"])
    fr = np.flatnonzero((rec.source.values == 1) & (rec.country.values == "France"))
    W = world_stats("test", rec.source.values == 1, f"all_{SET}")
    flags = load_flags("test")
    pr = joblib.load(work("train", f"pruner_{SET}.joblib"))
    mt = joblib.load(work("train", "matcher_run3.joblib"))
    strings = load_strings("test")
    out, t0 = [], time.time()
    for i, pairs in enumerate(candidate_chunks("test", fr, window=500_000)):
        df = matcher_view(full_features(prune(cheap_features(pairs, W, flags), pr), strings, rec.source.values), W)
        df["prob3"] = predict_chunked(mt["model"], df, mt["cols"])
        out.append(df)
        print(f"  France chunk {i+1}: {len(df):,} pairs, {time.time()-t0:.0f}s", flush=True)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(work("test", "france_features.parquet"), index=False)
    print(f"France pairs: {len(df):,} ({df.s1.nunique():,} S1)")


def pseudo_labels(df, all_scores):
    """all_scores: run3 probabilities of every test pair (so competing claims from any S1 are seen)."""
    best = all_scores.groupby("p").prob.max()
    s = all_scores.set_index(["s1", "p"]).prob
    other_best = (all_scores.assign(r=all_scores.groupby("p").prob.rank(ascending=False, method="first"))
                  .query("r == 2").set_index("p").prob)
    p_best = best.reindex(df.p.values).values
    p_second = other_best.reindex(df.p.values).fillna(0).values
    mine = df.prob3.values
    beaten = (mine < p_best) & (p_best >= 0.9)  # another S1 claims this record more strongly
    uncontested = (mine >= p_best) & (p_second < 0.5)
    pos = (mine >= 0.995) & uncontested
    neg = (mine <= 0.05) | beaten
    lab = np.full(len(df), -1, np.int8)
    lab[neg] = 0
    lab[pos] = 1
    return lab


def step_train():
    rec = load_records("train", ["source", "country"])
    A, B, R = masks(rec)
    tr = load_trainset(rec, R)
    cols = joblib.load(work("train", "matcher_run3.joblib"))["cols"]
    fr = pd.read_parquet(work("test", "france_features.parquet"))
    sc = pd.read_parquet(work("test", "scores_run3.parquet"))
    lab = pseudo_labels(fr, sc)
    fr = fr[lab >= 0].assign(label=lab[lab >= 0].astype(bool))
    print(f"France pseudo-labels: {len(fr):,} pairs ({fr.label.mean():.3f} positive); "
          f"decoy-type negatives: {((fr.prob3 > 0.05) & ~fr.label).sum():,}", flush=True)
    X = np.vstack([tr[cols].to_numpy(np.float32), fr[cols].to_numpy(np.float32)])
    y = np.r_[tr.label.values, fr.label.values]
    del tr
    m = make_model({"max_iter": 800})
    t = time.time()
    m.fit(X, y)
    joblib.dump({"model": m, "cols": cols}, work("train", "matcher_run4.joblib"))
    print(f"matcher_run4: {m.n_iter_} iters, {time.time()-t:.0f}s")


def step_score():
    fr = pd.read_parquet(work("test", "france_features.parquet"))
    mt = joblib.load(work("train", "matcher_run4.joblib"))
    fr["prob"] = predict_chunked(mt["model"], fr, mt["cols"])
    sc = pd.read_parquet(work("test", "scores_run3.parquet"))
    rec = load_records("test", ["country"])
    keep = sc[rec.country.values[sc.s1.values] != "France"]
    new = pd.concat([keep, fr[["s1", "p", "prob"]]], ignore_index=True)
    assert len(new) == len(sc)
    new.to_parquet(work("test", "scores_run4.parquet"), index=False)
    d = fr[["s1", "p", "prob3", "prob"]]
    print("France: pairs >=0.95 run3:", (d.prob3 >= 0.95).sum(), " run4:", (d.prob >= 0.95).sum())
    print("France prob change histogram (run4 - run3):", np.histogram(d.prob - d.prob3, bins=[-1, -.5, -.2, -.05, .05, .2, .5, 1])[0])


if __name__ == "__main__":
    {"features": step_features, "train": step_train, "score": step_score}[sys.argv[1]]()
