"""Raw TSV -> Parquet conversion. One `records.parquet` per split holding all three sources.

Row position in records.parquet is the record's integer index (`idx`) used everywhere downstream.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .paths import DATA_DIR, work

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pa.Table:
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in COLS + ["source1_entity_id", "matched_entity_ids"]},
                                             strings_can_be_null=False),
    )


def prepare(split: str) -> None:
    frames = []
    for s in (1, 2, 3):
        t = read_tsv(DATA_DIR / split / f"{split}_source{s}.tsv")
        assert t.column_names == COLS, t.column_names
        df = t.to_pandas()
        df["source"] = np.int8(s)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"business_name": "name", "business_address": "address"})
    assert df.entity_id.is_unique
    df["country"] = df["country"].astype("category")
    df.to_parquet(work(split, "records.parquet"), index=False)

    if split == "train":
        gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv").to_pandas()
        idx = pd.Series(np.arange(len(df), dtype=np.int32), index=df.entity_id)
        rows = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
        rows = rows[rows.m.notna() & (rows.m != "")]
        pairs = pd.DataFrame({"s1": idx.loc[rows.source1_entity_id].values, "p": idx.loc[rows.m].values})
        pairs.to_parquet(work(split, "gt_pairs.parquet"), index=False)


def load_records(split: str, columns=None) -> pd.DataFrame:
    return pd.read_parquet(work(split, "records.parquet"), columns=columns)


def load_gt_pairs() -> pd.DataFrame:
    return pd.read_parquet(work("train", "gt_pairs.parquet"))


if __name__ == "__main__":
    import sys, time
    for split in sys.argv[1:] or ["train", "test"]:
        t = time.time()
        prepare(split)
        print(split, "prepared in", round(time.time() - t, 1), "s")
