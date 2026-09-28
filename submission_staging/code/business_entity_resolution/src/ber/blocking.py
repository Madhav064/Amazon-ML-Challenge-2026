"""Stage 2: candidate generation by sparse TF-IDF top-K retrieval, one retrieval "channel" per term family.

For each country (records never match across countries), every S1 record queries an inverted index
built over S2+S3 records of that country. Terms are hashed to int64 inside worker processes so the
vocabulary never lives in Python objects. Very frequent terms (pool df > cap) are skipped: they carry
almost no IDF weight and dominate cost.
"""
import time
from multiprocessing import Pool
from zlib import crc32

import numba as nb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .io import load_records
from .paths import work


# ----------------------------------------------------------------------------- term families
def _name_terms(nc, sk, nk):
    return ["n" + t for t in nc.split()] + ["s" + t for t in sk.split()]


def _addr_terms(an):
    toks = an.split()
    return ["a" + t for t in toks] + ["b" + a + "_" + b for a, b in zip(toks, toks[1:])]


def _cgram_terms(nk, n=4):
    s = "^" + nk + "$"
    if len(s) <= n:
        return ["c" + s]
    return ["c" + s[i:i + n] for i in range(len(s) - n + 1)]


CHANNELS = {
    "name": (("nc", "sk", "nk"), lambda r: _name_terms(*r)),
    "addr": (("an",), lambda r: _addr_terms(r[0])),
    "cgram": (("nk",), lambda r: _cgram_terms(r[0])),
    "joint": (("nc", "sk", "nk", "an"), lambda r: _name_terms(r[0], r[1], r[2]) + _addr_terms(r[3])),
    "name_noaddr": (("nc", "sk", "nk"), lambda r: _name_terms(*r)),
    # pool restricted to native-script names: phonetic-skeleton name terms + address terms
    "native": (("sk", "an"), lambda r: ["s" + t for t in r[0].split()] + _addr_terms(r[1])),
    # pool restricted to empty-address records: character 4-grams catch typos in the name
    "cgram_noaddr": (("nk",), lambda r: _cgram_terms(r[0])),
}


def _hash(t: str) -> int:
    b = t.encode()
    return (crc32(b) << 32) | crc32(b, 0x9E3779B9)


def _terms_part(args):
    path, channel = args
    fn = CHANNELS[channel][1]
    need = list(CHANNELS[channel][0])
    f = pq.ParquetFile(path)
    L, H, W = [], [], []
    for b in f.iter_batches(batch_size=100_000, columns=sorted(set(need) | {"nc", "nk", "web", "an", "native"})):
        cols = b.to_pandas()
        lens, hs = [], []
        for row in zip(*[cols[c].tolist() for c in need]):
            ts = set(fn(row))
            lens.append(len(ts))
            hs.extend(_hash(t) for t in ts)
        L.append(np.array(lens, np.int32))
        H.append(np.array(hs, np.uint64).view(np.int64))
        W.append(np.stack([cols.web.values | (~cols.nc.str.contains(" ", regex=False).values & (cols.nk.str.len().values >= 8)),
                           (cols.an == "").values, cols.native.values], axis=1))
    return np.concatenate(L), np.concatenate(H), np.concatenate(W)


def build_terms(split: str, channel: str, workers=6):
    parts = sorted(work(split, "norm").glob("part-*.parquet"))
    with Pool(min(workers, len(parts))) as pool:
        res = pool.map(_terms_part, [(p, channel) for p in parts])
    lens = np.concatenate([r[0] for r in res])
    hs = np.concatenate([r[1] for r in res])
    webish = np.concatenate([r[2] for r in res])
    ptr = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=ptr[1:])
    return ptr, hs, webish


# ----------------------------------------------------------------------------- retrieval kernel
@nb.njit(parallel=True, cache=True)
def _topk(q_ptr, q_term, q_w, q_order, p_ptr, p_term, p_w, i_ptr, i_doc, i_w, n_docs, K, M, cap, min_terms, nblocks):
    """Phase 1: accumulate partial scores over the query's rare terms (pool df <= cap, plus always its
    `min_terms` rarest terms) and shortlist the top-M docs. Phase 2: exact cosine for the shortlist by
    merge-joining term-sorted rows; keep top-K. Rows of q_term/p_term are sorted by term id;
    q_order lists each query's term positions from rarest to most frequent."""
    nq = q_ptr.shape[0] - 1
    out_d = np.full((nq, K), -1, np.int32)
    out_s = np.zeros((nq, K), np.float32)
    step = (nq + nblocks - 1) // nblocks
    for b in nb.prange(nblocks):
        lo = b * step
        hi = min(nq, lo + step)
        acc = np.zeros(n_docs, np.float32)
        touched = np.empty(n_docs, np.int32)
        for q in range(lo, hi):
            nt = 0
            used = 0
            for jj in range(q_ptr[q], q_ptr[q + 1]):
                j = q_order[jj]
                t = q_term[j]
                s = i_ptr[t]
                e = i_ptr[t + 1]
                if e == s:
                    continue
                if e - s > cap and used >= min_terms:
                    continue
                used += 1
                wq = q_w[j]
                for k in range(s, e):
                    d = i_doc[k]
                    if acc[d] == 0.0:
                        touched[nt] = d
                        nt += 1
                    acc[d] += wq * i_w[k]
            if nt == 0:
                continue
            sc = np.empty(nt, np.float32)
            for i in range(nt):
                sc[i] = -acc[touched[i]]
                acc[touched[i]] = 0.0
            if nt > M:  # O(nt) selection of the M best partial scores (ties beyond M dropped)
                kth = np.partition(sc.copy(), M - 1)[M - 1]
                short = np.empty(M, np.int32)
                m = 0
                for i in range(nt):
                    if sc[i] < kth and m < M:
                        short[m] = touched[i]
                        m += 1
                for i in range(nt):
                    if sc[i] == kth and m < M:
                        short[m] = touched[i]
                        m += 1
            else:
                m = nt
                short = touched[:nt].copy()
            ex = np.zeros(m, np.float32)
            qs, qe = q_ptr[q], q_ptr[q + 1]
            for r in range(m):
                d = short[r]
                a, ae = qs, qe
                c, ce = p_ptr[d], p_ptr[d + 1]
                tot = 0.0
                while a < ae and c < ce:
                    ta, tc = q_term[a], p_term[c]
                    if ta == tc:
                        tot += q_w[a] * p_w[c]
                        a += 1
                        c += 1
                    elif ta < tc:
                        a += 1
                    else:
                        c += 1
                ex[r] = tot
            order = np.argsort(-ex)
            for r in range(min(m, K)):
                out_d[q, r] = short[order[r]]
                out_s[q, r] = ex[order[r]]
    return out_d, out_s


@nb.njit(cache=True)
def _invert(p_term, p_doc, p_w, V):
    """Counting sort of postings by term (docs stay in increasing order within a term)."""
    cnt = np.zeros(V + 1, np.int64)
    for t in p_term:
        cnt[t + 1] += 1
    ptr = np.cumsum(cnt)
    cur = ptr[:-1].copy()
    doc = np.empty(len(p_term), np.int32)
    w = np.empty(len(p_term), np.float32)
    for i in range(len(p_term)):
        t = p_term[i]
        j = cur[t]
        doc[j] = p_doc[i]
        w[j] = p_w[i]
        cur[t] = j + 1
    return ptr, doc, w


@nb.njit(parallel=True, cache=True)
def _sort_rows(ptr, term, w):
    """Sort every CSR row by term id, in place."""
    for r in nb.prange(len(ptr) - 1):
        a, b = ptr[r], ptr[r + 1]
        if b - a > 1:
            o = np.argsort(term[a:b])
            term[a:b] = term[a:b][o]
            w[a:b] = w[a:b][o]


@nb.njit(parallel=True, cache=True)
def _row_order(ptr, key):
    """Global positions of each row's entries, ordered by ascending key within the row."""
    out = np.empty(len(key), np.int64)
    for r in nb.prange(len(ptr) - 1):
        a, b = ptr[r], ptr[r + 1]
        o = np.argsort(key[a:b], kind="mergesort")
        for i in range(b - a):
            out[a + i] = a + o[i]
    return out


def _subset_csr(ptr, hs, rows):
    lens = (ptr[rows + 1] - ptr[rows]).astype(np.int64)
    sub_ptr = np.zeros(len(rows) + 1, np.int64)
    np.cumsum(lens, out=sub_ptr[1:])
    idx = np.repeat(ptr[rows] - sub_ptr[:-1], lens) + np.arange(sub_ptr[-1])
    return sub_ptr, hs[idx]


def retrieve(ptr, hs, q_rows, p_rows, K, cap, nblocks=12, M_mult=4, min_terms=2, q_all_rows=None):
    """Top-K pool rows for each query row by exact IDF-weighted cosine over binary term vectors.
    IDF is computed over q_all_rows (every S1 record of the country) + pool, so scores do not depend
    on which subset of queries is run."""
    t0 = time.time()
    q_all_rows = q_rows if q_all_rows is None else q_all_rows
    qa_ptr, qa_h = _subset_csr(ptr, hs, q_all_rows)
    p_ptr, p_h = _subset_csr(ptr, hs, p_rows)
    nq_terms = len(qa_h)
    uniq, inv = np.unique(np.concatenate([qa_h, p_h]), return_inverse=True)
    del qa_h, p_h
    V = len(uniq)
    del uniq
    inv = inv.astype(np.int32)
    df = np.bincount(inv, minlength=V)
    idf = np.log1p((len(q_all_rows) + len(p_rows)) / df).astype(np.float32)
    del df
    w = idf[inv]
    qa_term, p_term = inv[:nq_terms], inv[nq_terms:]
    qa_w, p_w = w[:nq_terms], w[nq_terms:]
    for p_, w_ in ((qa_ptr, qa_w), (p_ptr, p_w)):
        lens = np.diff(p_)
        nz = lens > 0
        sq = np.zeros(len(lens), np.float32)
        sq[nz] = np.add.reduceat(w_ * w_, p_[:-1][nz])
        w_ /= np.repeat(np.sqrt(np.maximum(sq, 1e-12)), lens)
    pos = np.searchsorted(q_all_rows, q_rows)
    q_ptr, q_term = _subset_csr(qa_ptr, qa_term, pos)
    _, q_w = _subset_csr(qa_ptr, qa_w, pos)
    del qa_term, qa_w
    p_doc = np.repeat(np.arange(len(p_rows), dtype=np.int32), np.diff(p_ptr))
    i_ptr, i_doc, i_w = _invert(p_term, p_doc, p_w, V)
    del p_doc
    df_pool = np.diff(i_ptr).astype(np.int32)
    # term-sorted rows for the exact phase-2 merge join
    p_term, p_w, q_term, q_w = p_term.copy(), p_w.copy(), q_term.copy(), q_w.copy()
    _sort_rows(p_ptr, p_term, p_w)
    _sort_rows(q_ptr, q_term, q_w)
    q_order = _row_order(q_ptr, df_pool[q_term])
    del df_pool
    t1 = time.time()
    d, s = _topk(q_ptr, q_term, q_w, q_order, p_ptr, p_term, p_w, i_ptr, i_doc, i_w, len(p_rows),
                 K, K * M_mult, cap, min_terms, nblocks)
    print(f"    retrieve {len(q_rows):,} q x {len(p_rows):,} docs, {V/1e6:.1f}M terms: "
          f"index {t1-t0:.0f}s, search {time.time()-t1:.0f}s", flush=True)
    rank = np.tile(np.arange(K, dtype=np.int16), len(q_rows))
    d, s = d.ravel(), s.ravel()
    keep = d >= 0
    return pd.DataFrame({
        "s1": np.repeat(q_rows.astype(np.int32), K)[keep],
        "p": p_rows.astype(np.int32)[d[keep]],
        "score": s[keep],
        "rank": rank[keep],
    })


# ----------------------------------------------------------------------------- driver
def query_mask(split: str, rec: pd.DataFrame, sample: float) -> np.ndarray:
    """S1 rows to query. sample < 1 uses a fixed random subset (dev mode; queries are independent,
    so recall measured on the subset is an unbiased estimate of full recall)."""
    q = rec.source.values == 1
    if sample < 1:
        rng = np.random.default_rng(0)
        q &= rng.random(len(rec)) < sample
    return q


def cand_path(split, channel, sample):
    return work(split, f"cand_{channel}.parquet" if sample >= 1 else f"cand_{channel}_s{int(sample * 100)}.parquet")


def run_reverse(split: str, channel: str, K: int = 5, cap: int = 5_000) -> pd.DataFrame:
    """Reverse retrieval: every S2/S3 record queries an index of the S1 records of its country and keeps
    its top-K S1. Recovers records whose true S1 is crowded out of that S1's forward top-K list by
    same-name look-alikes. Output uses the forward orientation (s1, p, score, rank-within-the-record)."""
    t0 = time.time()
    base = channel[:-4]
    rec = load_records(split, ["source", "country"])
    ptr, hs, _ = build_terms(split, base)
    t1 = time.time()
    outs = []
    for c in rec.country.cat.categories:
        cm = (rec.country == c).values
        pool = np.flatnonzero(cm & (rec.source.values > 1))
        s1 = np.flatnonzero(cm & (rec.source.values == 1))
        if len(pool) == 0 or len(s1) == 0:
            continue
        r = retrieve(ptr, hs, pool, s1, K, cap, q_all_rows=pool)
        outs.append(r.rename(columns={"s1": "p", "p": "s1"})[["s1", "p", "score", "rank"]])
    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(cand_path(split, channel, 1.0), index=False)
    print(f"[{split}/{channel}] terms {len(hs)/1e6:.0f}M in {t1-t0:.0f}s, retrieval {time.time()-t1:.0f}s, "
          f"{len(out)/1e6:.1f}M pairs", flush=True)
    return out


def run_channel(split: str, channel: str, K: int = 50, cap: int = 5_000, sample: float = 1.0) -> pd.DataFrame:
    if channel.endswith("_rev"):
        return run_reverse(split, channel, K, cap)
    t0 = time.time()
    rec = load_records(split, ["source", "country"])
    ptr, hs, flags = build_terms(split, channel)
    t1 = time.time()
    pm = rec.source.values > 1
    if channel == "cgram":  # char-gram channel only indexes concatenated / web-style pool names
        pm &= flags[:, 0]
    if channel in ("name_noaddr", "cgram_noaddr"):  # pool records whose address is empty
        pm &= flags[:, 1]
    if channel == "native":  # pool records whose name is in an Indic script
        pm &= flags[:, 2]
    qm = query_mask(split, rec, sample)
    outs = []
    for c in rec.country.cat.categories:
        cm = (rec.country == c).values
        q = np.flatnonzero(cm & qm)
        p = np.flatnonzero(cm & pm)
        if len(q) == 0 or len(p) == 0:
            continue
        outs.append(retrieve(ptr, hs, q, p, K, cap, q_all_rows=np.flatnonzero(cm & (rec.source.values == 1))))
    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(cand_path(split, channel, sample), index=False)
    print(f"[{split}/{channel}] terms {len(hs)/1e6:.0f}M in {t1-t0:.0f}s, retrieval {time.time()-t1:.0f}s, "
          f"{len(out)/1e6:.1f}M pairs ({len(out) / qm.sum():.1f}/S1)", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("channels", nargs="+")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--cap", type=int, default=5_000)
    ap.add_argument("--sample", type=float, default=1.0)
    a = ap.parse_args()
    for ch in a.channels:
        run_channel(a.split, ch, a.K, a.cap, a.sample)
