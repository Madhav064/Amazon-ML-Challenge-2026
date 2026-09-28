# Business Entity Resolution — reproducible pipeline

Normalization → 7 forward + 1 reverse sparse TF-IDF retrieval channels → learned pruner (≈7 candidates per Source-1 entity)
→ gradient-boosted pair matcher → exclusivity + per-country cutoff →
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Only the competition data is used: no external data, APIs, geocoding or pretrained models. All
normalization rules (Indic-script transliteration, abbreviations, legal forms) are hand-written in
`src/ber/text.py`. Models are scikit-learn `HistGradientBoostingClassifier` (BSD-3, a few MB).

## Environment

Python 3.10, CPU only. Developed on 12 threads / 15 GB RAM / Windows 11; every step is chunked to fit
in ~7 GB of free RAM.

```bash
pip install -r requirements.txt
```

Data is read from `../student_resource/dataset` (override with `BER_DATA_DIR`); intermediate files go
to `../work` (`BER_WORK_DIR`, ~20 GB peak); outputs to `../output` (`BER_OUTPUT_DIR`).

## Reproduce the submission (run from `src/`)

Approximate wall-clock times on the machine above in brackets.

```bash
# 1. data -> parquet, normalization                                        [~10 min]
python -m ber.io train test
python -m ber.normalize train test

# 2. candidate retrieval, 7 channels, every Source-1 record queried        [~35 min per split]
python -m ber.blocking train joint --K 30
python -m ber.blocking train addr --K 20
python -m ber.blocking train name cgram name_noaddr native cgram_noaddr --K 10
python -m ber.blocking train joint_rev --K 5    # reverse: each S2/S3 record's best S1
python -m ber.blocking test joint --K 30
python -m ber.blocking test addr --K 20
python -m ber.blocking test name cgram name_noaddr native cgram_noaddr --K 10
python -m ber.blocking test joint_rev --K 5

# 3. pruner (sample A) and matcher training set (sample B, pruned)         [~40 min]
python -m ber.cascade pruner
python -m ber.cascade trainset
python -m ber.cascade cv        # optional: 2-fold grouped validation, prints macro F0.5 [~25 min]

# 4. final matcher: sample B + sample C (~1.7x more entities)              [~40 min]
python -m ber.run5 trainset
python -m ber.run5 final
python -m ber.run5 test         # prune + score every test candidate        [~65 min]

# 5. decisions -> output/  (exclusivity; cutoffs US/India 0.85, France 0.95)  [~3 min]
python -m ber.finalize run6

# 6. validate
cd ../../student_resource
python utils/validate_submission.py -m ../output/matching_results.tsv -c ../output/candidate_pairs.tsv -t dataset/test --check-ids
```

In the submission-zip layout set `BER_DATA_DIR` to the challenge `dataset/` folder (and optionally
`BER_WORK_DIR` / `BER_OUTPUT_DIR`) before running.

`ber.selftrain` (France self-training) is included for completeness: it was evaluated on the leaderboard
(0.9500 vs 0.9533) and is **not** part of the final submission.

## Module map

Final pipeline:

| module | role |
|---|---|
| `io.py` | TSV → parquet, integer record index, ground-truth pairs |
| `text.py` | normalization: unicode folding, transliteration of 9 Indic scripts, legal forms, DBA / web / junk stripping, address abbreviations, phonetic skeleton |
| `normalize.py` | parallel, streaming normalization of all records |
| `blocking.py` | numba TF-IDF top-K retrieval (channels `joint`, `addr`, `name`, `cgram`, `name_noaddr`, `native`, `cgram_noaddr`, and reverse `joint_rev`) |
| `features.py` | candidate union, pairwise string / number features |
| `cascade.py` | test-like simulation, rarity & competition statistics, pruner, matcher training/validation, test scoring |
| `run5.py` | final matcher: training set C, training on B + C, test scoring |
| `finalize.py` | exclusivity + per-country cutoffs, writes both output files |
| `decide.py`, `metrics.py` | decision rules; exact competition metric (per-entity F0.5, macro average incl. singletons) |

Development / analysis tools (used for the experiments reported in the documentation):
`experiment.py` (config-driven grouped-CV experiments, `configs/*.json`, log in `experiments/results.jsonl`),
`blocking_eval.py` (recall and oracle F0.5 of a candidate set), `analysis.py` (FP/FN breakdown),
`stack.py` (stage-2 stacking experiment), `selftrain.py` (France self-training, rejected), `pipeline.py` (first full-scale scoring path), `shift.py`
(train/test feature-shift check).
