"""Stage 1: normalize every record of a split -> work/<split>/norm/part-XX.parquet (row-aligned with records).

Each worker streams one parquet row group in small batches and writes its own part file, so no
process ever holds more than ~100k rows of Python strings (the machine has ~7 GB usable RAM).
"""
import re
import time
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .paths import work
from .text import _RE_INDIC, normalize_address, normalize_name, skeleton

_RE_WEB = re.compile(r"(?i)(\.com|\.in\b|\.net|\.org|www\.|^@)")
NORM_COLS = ["nn", "nc", "nk", "sk", "an", "num", "native", "web"]


def _normalize_batch(names, addrs):
    cols = {k: [] for k in NORM_COLS}
    for n, a in zip(names, addrs):
        full, core, compact = normalize_name(n)
        ta, tn = normalize_address(a)
        cols["nn"].append(full); cols["nc"].append(core); cols["nk"].append(compact)
        cols["sk"].append(" ".join(skeleton(t) for t in core.split()))
        cols["an"].append(ta); cols["num"].append(tn)
        cols["native"].append(bool(_RE_INDIC.search(n)))
        cols["web"].append(bool(_RE_WEB.search(n)))
    return pa.table(cols)


def _row_group(args):
    split, rg = args
    f = pq.ParquetFile(work(split, "records.parquet"))
    t = f.read_row_group(rg, columns=["name", "address"])
    out = work(split, "norm", f"part-{rg:03d}.parquet")
    writer = None
    for b in t.to_batches(max_chunksize=100_000):
        tbl = _normalize_batch(b.column(0).to_pylist(), b.column(1).to_pylist())
        writer = writer or pq.ParquetWriter(out, tbl.schema)
        writer.write_table(tbl)
    writer.close()
    return rg


def run(split: str, workers: int = 11) -> None:
    t0 = time.time()
    work(split, "norm").mkdir(exist_ok=True)
    n_rg = pq.ParquetFile(work(split, "records.parquet")).metadata.num_row_groups
    with Pool(min(workers, n_rg)) as pool:
        list(pool.imap_unordered(_row_group, [(split, i) for i in range(n_rg)]))
    print(f"normalized {split} in {time.time() - t0:.0f}s")


def load_norm(split: str, columns=None) -> pd.DataFrame:
    d = work(split, "norm")
    parts = sorted(d.glob("part-*.parquet"))
    return pa.concat_tables([pq.read_table(p, columns=columns) for p in parts]).to_pandas()


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or ["train", "test"]:
        run(s)
