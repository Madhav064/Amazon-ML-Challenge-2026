"""Final decision rule -> output/matching_results.tsv + output/candidate_pairs.tsv.

Exclusivity (each S2/S3 record goes to at most one S1: the one with the highest probability), then a
per-country probability cutoff. US/India 0.85; France 0.95 -- France is absent from training, and on
the test set its mid-probability matches are far more often claimed more strongly by another S1
(look-alike conflicts: ~15% vs <1% for US/India), so it gets a stricter cutoff. Countries not listed
use DEFAULT.

    python -m ber.finalize <scores name, e.g. run4>
"""
import sys

import numpy as np
import pandas as pd

from . import decide
from .io import load_records
from .paths import OUTPUT_DIR, work

CUTOFFS = {"US": 0.85, "India": 0.85, "France": 0.95}
DEFAULT = 0.85


def finalize(name, cutoffs=CUTOFFS, out_dir=OUTPUT_DIR):
    rec = load_records("test", ["entity_id", "source", "country"])
    ids = rec.entity_id.values
    s1_ids = np.flatnonzero(rec.source.values == 1)
    sc = pd.read_parquet(work("test", f"scores_{name}.parquet"))
    d = decide.exclusive(sc)
    thr = pd.Series(rec.country.values[d.s1.values]).map(cutoffs).fillna(DEFAULT).values
    pred = d[d.prob.values >= thr]
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame, col, fname in ((pred, "matched_entity_ids", "matching_results.tsv"),
                              (sc, "candidate_entity_ids", "candidate_pairs.tsv")):
        f = frame[["s1", "p"]].sort_values(["s1", "p"])
        lists = f.assign(pid=ids[f.p.values]).groupby("s1").pid.agg(",".join).reindex(s1_ids).fillna("")
        with open(out_dir / fname, "w", newline="\n", encoding="utf-8") as fh:
            fh.write(f"source1_entity_id\t{col}\n")
            for a, b in zip(ids[s1_ids], lists.values):
                fh.write(f"{a}\t{b}\n")
        print(f"wrote {out_dir / fname}: {len(f):,} ids ({len(f) / len(s1_ids):.2f} per S1)")


if __name__ == "__main__":
    finalize(sys.argv[1], out_dir=OUTPUT_DIR / sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR)
