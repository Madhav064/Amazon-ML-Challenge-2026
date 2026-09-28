"""Covariate-shift check: compute matcher features for one test window and compare against the train
set, on 'clean-looking' pairs (high name and address similarity) per country."""
import joblib
import numpy as np
import pandas as pd

from .cascade import candidate_chunks, cheap_features, full_features, load_flags, prune, world_stats, RUN, NON_FEAT
from .features import load_strings
from .io import load_records
from .paths import work

if __name__ == "__main__":
    rec = load_records("test", ["source", "country"])
    s1 = np.flatnonzero(rec.source.values == 1)
    rng = np.random.default_rng(0)
    s1 = np.sort(rng.choice(s1, 60_000, replace=False))
    W = world_stats("test", rec.source.values == 1, f"all_{RUN}")
    flags = load_flags("test")
    pr = joblib.load(work("train", f"pruner_{RUN}.joblib"))
    mt = joblib.load(work("train", f"matcher_{RUN}.joblib"))
    strings = load_strings("test")
    parts = [full_features(prune(cheap_features(p, W, flags), pr), strings, rec.source.values)
             for p in candidate_chunks("test", s1, window=2_000_000)]
    te = pd.concat(parts, ignore_index=True)
    te["country"] = rec.country.values[te.s1.values]
    te["prob"] = mt["model"].predict_proba(te[mt["cols"]].to_numpy(np.float32))[:, 1]
    te.to_parquet(work("test", "shift_sample.parquet"), index=False)
    tr = pd.read_parquet(work("train", f"trainset_B_{RUN}.parquet"))
    rtr = load_records("train", ["country"])
    tr["country"] = rtr.country.values[tr.s1.values]
    cols = [c for c in mt["cols"]]
    for c in ["India", "US"]:
        a = tr[(tr.country == c) & (tr.nc_tset >= 0.9) & (tr.an_tset >= 0.8)]
        b = te[(te.country == c) & (te.nc_tset >= 0.9) & (te.an_tset >= 0.8)]
        print(f"\n=== {c}: clean-looking pairs  train n={len(a):,} label-rate={a.label.mean():.3f}  |  test n={len(b):,}  "
              f"mean prob test={b.prob.mean():.3f}")
        rows = []
        for col in cols:
            x, y = a[col].astype(float), b[col].astype(float)
            sd = np.nanstd(np.r_[x, y]) + 1e-9
            rows.append((col, np.nanmedian(x), np.nanmedian(y), (np.nanmean(y) - np.nanmean(x)) / sd))
        d = pd.DataFrame(rows, columns=["feature", "train_median", "test_median", "std_shift"])
        print(d.reindex(d.std_shift.abs().sort_values(ascending=False).index).head(14).round(3).to_string(index=False))
