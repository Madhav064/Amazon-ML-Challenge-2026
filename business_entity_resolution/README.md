# Business Entity Resolution — reproducible pipeline

Blocking (sparse TF-IDF retrieval, 5 channels) → pairwise features → gradient-boosted pair classifier →
F0.5-tuned decisions → `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

Only the competition data is used. No external data, APIs, geocoding or pretrained models; every
normalization rule (Indic-script transliteration, abbreviations, legal forms) is hand-written in
`src/ber/text.py`.

## Environment

Python 3.10, CPU only (developed on 12 threads / 15 GB RAM / Windows).

```bash
pip install -r requirements.txt
```

Data is expected at `../student_resource/dataset` (override with `BER_DATA_DIR`). Intermediate
files go to `../work` (`BER_WORK_DIR`, ~15 GB peak), outputs to `../output` (`BER_OUTPUT_DIR`).

## Reproduce end to end

Run from `src/`; `configs/final.json` fixes blocking depths, features, model and decision rule.

```bash
python -m ber.io train test                                   # TSV -> parquet (+ ground-truth pairs)
python -m ber.normalize train test                            # name/address normalization
python -m ber.pipeline block train ../configs/final.json      # candidate retrieval, all channels
python -m ber.pipeline block test  ../configs/final.json
python -m ber.pipeline train ../configs/final.json            # fit the pair classifier (train dev sample)
python -m ber.pipeline score ../configs/final.json test       # features + probabilities for test candidates
python -m ber.pipeline write ../configs/final.json            # decisions -> output/*.tsv
python ../../student_resource/utils/validate_submission.py -m ../../output/matching_results.tsv \
       -c ../../output/candidate_pairs.tsv -t ../../student_resource/dataset/test
```

Validation (optional):

```bash
python -m ber.experiment ../configs/<cfg>.json                # grouped-CV OOF experiment on the dev sample
python -m ber.analysis <experiment_name> --t 0.7 --exclusive  # FP / FN breakdown with examples
python -m ber.pipeline score ../configs/final.json holdout    # score the 90% of train S1 never trained on
python -m ber.pipeline holdout ../configs/final.json          # full-competition macro F0.5 per threshold
python -m ber.blocking_eval joint addr name cgram name_noaddr # blocking recall / oracle F0.5
```

## Layout

| module | stage |
|---|---|
| `io.py` | TSV → parquet, integer record index, ground-truth pairs |
| `text.py` | normalization: unicode folding, Indic transliteration, legal forms, DBA/web/junk stripping, address abbreviations, phonetic skeleton |
| `normalize.py` | parallel normalization of all records |
| `blocking.py` | numba TF-IDF top-K retrieval, channels `joint`, `addr`, `name`, `cgram`, `name_noaddr` |
| `blocking_eval.py` | recall@K, candidates/S1, oracle F0.5 of a candidate set |
| `features.py` | candidate union + pairwise, rarity, house-number and pool-competition features |
| `experiment.py` | config-driven grouped-CV experiments, decision grid, results log (`experiments/results.jsonl`) |
| `decide.py` | exclusivity, threshold and expected-F0.5 decision rules |
| `metrics.py` | exact competition metric (per-S1 F0.5, macro average incl. singletons) |
| `analysis.py` | false-positive / false-negative analysis |
| `pipeline.py` | full-scale scoring, holdout evaluation, output writing |
