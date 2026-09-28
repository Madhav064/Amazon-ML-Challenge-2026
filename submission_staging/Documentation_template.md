# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Tetris
**Team Members:** Madhav Raj, Kapil Meena, Devesh Sarda, Ishika Kanyal
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve Source-2/3 records to the deduplicated Source-1 reference with a three-stage cascade: seven
forward sparse TF-IDF retrieval channels plus one reverse channel (numba, exact cosine re-scoring) generate
~67 candidates per Source-1 entity (98.2 % pair recall); a learned pruner trained only on retrieval-stage
signals cuts this to **7.3 candidates per entity** (97.5 % pair recall kept); a gradient-boosted pair matcher with ~100 name, address,
house-number, rarity and "competition" features scores the survivors, and an exclusivity rule plus a
per-country probability cutoff produces the matches. Key ideas: script-aware normalization (our own
transliteration of 9 Indic scripts), channels targeted at the hard cases (native-script names,
empty addresses, web-style names), **reverse retrieval** (every S2/S3 record also looks up its best
Source-1 entities), a train-side simulation of the test's higher density of unmatched
look-alike records, shift-robust features, and label-free, conflict-based cutoffs for France, which
appears only in the test set. Public leaderboard: 0.9426 → **0.963** over our submissions.

---

## 2. Methodology

### 2.1 Problem Analysis

Profiling of all 23.4 M records (train 12.5 M, test 11.7 M):

- **Scale.** Train: 2.2 M S1, 5.0 M S2, 5.3 M S3; test: 1.7 M S1, 4.9 M S2, 5.1 M S3 → all-pairs is ~10¹³, so
  blocking is mandatory. All files are clean UTF-8; IDs unique; no train/test overlap.
- **Ground truth.** 3.46 matches per S1 on average (max 11: ≤5 from S2, ≤6 from S3); **5.6 % singletons**.
  Every S2/S3 record belongs to **at most one** S1 (a partition) and matches are always within the same
  country. ~26 % of S2/S3 records match no S1 (distractors).
- **Names.** Heavy reuse: 38 % of S1 rows share their exact name with another S1 at a different address
  ("Primary Care Group" ×253), so a name alone rarely identifies an entity. Only 4.7 % of true pairs have
  identical raw names (25 % after normalization); 14 % share no name token at all. Noise: legal-form swaps
  (Pvt/Private, LLC/L.L.C., SARL/SAS), word order, typos and digit/letter swaps (`Fashi0n`, `PRlVATE`),
  honorifics (Mr, Smt, M/s), `X DBA Y` / `trading as` (the true name is Y), `(ID: 1728)` tags, leading
  junk, **web-style names** (`rkkproducts.com`, `@villalvazofederal`, 4 %), and **native-script names**
  for India (28 % of S2 India names; Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam,
  Oriya, Gurmukhi).
- **Addresses.** 3.4 % of S2/S3 addresses empty (and empty addresses are 15× more common among matched
  records than unmatched ones); `NULL`/`N/A` placeholders; US component reordering; India addresses long
  with landmarks and house-number formats (`H.No.16-11-23/37/A`); PIN codes essentially absent; state
  names in native script or as codes (GJ, TN); **house numbers perturbed** both in true matches
  (54 vs 54-56) and in look-alike decoys (5116 vs 5121 on the same street).
- **Train/test differences** (found during the challenge, see §5): France (15 % of test S1) never
  appears in training; the test has 5.75 S2/S3 records per S1 vs 4.67 in train; US has half as many S1
  in test as in train; French and test-India neighbourhoods are much more crowded with look-alikes
  (12.3 strong look-alike candidates per French S1 vs 3–4 for US/India).

### 2.2 Solution Strategy

**Approach Type:** Blocking (retrieval) + learned pruning + pairwise classifier + constrained decisions (cascade)
**Core Innovation:** multi-channel exact-cosine retrieval with channels for the hard cases, a
retrieval-signal pruner that cuts the candidate set ~10× before the expensive matcher, and features /
simulation designed so that a model trained on US/India transfers to a denser, partly unseen test set.

Validation always groups by Source-1 entity (no entity contributes pairs to both sides of a fold) and
uses the exact competition metric (`metrics.py`: per-S1 F0.5 = 1.25·tp / (0.25·n_true + n_pred), empty
prediction on a singleton = 1.0, macro average over all S1 including singletons and entities whose
matches were lost in blocking).

---

## 3. Candidate Generation (Blocking)

**Normalization** (`text.py`, applied identically to every source): NFKD accent stripping; our own
transliteration of the nine Indic Unicode blocks (they share the ISCII layout, so one offset table with
inherent-vowel / virama / nukta handling covers all); DBA resolution, removal of `(ID: n)`, web domains,
honorifics and leading junk; digit/letter-swap repair; canonical legal forms (`private→pvt`,
`limited→ltd`, `corporation→corp`, French SARL/SAS/EURL…); "core" name without legal forms and
stopwords; "compact" name without spaces (for web-style names); a phonetic consonant skeleton robust to
vowel spelling, transliteration and voicing (`एपेक्स पावर` → `apks pvr` = `apex power`); address
abbreviations (St/Street/Saint, Rd, Ave, Blvd, Rue/R, Bd…), placeholder removal and house-number
extraction without leading zeros.

**Blocking keys used — seven forward and one reverse sparse TF-IDF retrieval channel** (`blocking.py`). For every S1 record
(query) and every S2/S3 record of the same country (index), binary term vectors are IDF-weighted and
L2-normalized; each S1 retrieves its top-K pool records. The numba kernel accumulates partial scores
over the query's rare terms (pool document frequency ≤ 5 000, plus always its 2 rarest terms), keeps a
shortlist of 4·K by O(n) selection, then **re-scores the shortlist with the exact full cosine**
(merge-join of term-sorted rows) — skipping frequent terms without re-scoring biased the ranking
towards short names. IDF is computed over all S1 + pool records of the country, independent of which
queries are run.

| channel | terms | pool indexed | K |
|---|---|---|---|
| `joint` | core-name tokens + phonetic tokens + address tokens + address bigrams | all S2/S3 | 30 |
| `addr` | address tokens + bigrams (`97_liberty`) | all | 20 |
| `name` | core-name + phonetic tokens | all | 10 |
| `cgram` | character 4-grams of the compact name | web-style / concatenated names | 10 |
| `name_noaddr` | core-name + phonetic tokens | records with empty address | 10 |
| `native` | phonetic name tokens + address tokens | records with native-script names | 10 |
| `cgram_noaddr` | character 4-grams of the compact name | records with empty address | 10 |
| `joint_rev` (reverse) | as `joint`, but each **S2/S3 record** queries an index of S1 | all S1 | 2 per record |

The restricted-pool channels exist because error analysis showed the misses of the general channels
concentrate on empty-address records (37 % of misses vs 4 % of pairs), native-script names (29 % vs 7 %)
and web-style names: restricting the index removes the thousands of same-name competitors that push
those records out of a general top-K.

**Reverse retrieval.** Forward lists are per S1, so an S1 with many same-name look-alikes (chains, generic
names) fills its top-K with other records and its own true record can fall off. In the reverse channel every
S2/S3 record retrieves its own best S1 entities (same terms and IDF, so the score equals the joint cosine),
and the pair is added to that S1's candidates. Keeping each record's top 1 alone raised pair recall from
0.951 to 0.965 (India) and 0.978 to 0.991 (US) for only +0.3 candidates per S1; we keep the top 2.

**Learned pruner** (`cascade.py`). A small gradient-boosted model on retrieval-stage signals only
(channel scores and ranks, name/address rarity, pool-side competition margins, record flags; no string
similarity) scores each retrieved pair; a pair survives if its probability ≥ 0.01, at most 12 per S1.
The survivors are **exactly** the pairs the matcher scores, i.e. `candidate_pairs.tsv`.

**Candidate pairs generated:** test: ~116 M retrieved pairs (≈67 per S1) → **12.64 M pruned candidate
pairs, 7.30 per Source-1 entity** (261 of 1.73 M S1 have no candidate). Our first submission sent 33 per S1.

**How we ensured true matches were not lost** (measured on training ground truth):

| stage | pair recall | candidates / S1 |
|---|---|---|
| name channel alone @50 | 0.500 | 50 |
| address channel alone @50 | 0.885 | 50 |
| joint channel @20 | 0.937 | 20 |
| 5 general channels (first submissions) | 0.957 | 32.7 |
| 7 channels, deeper lists | 0.967 | ≈65 |
| after pruning (Run 2) | 0.961 | 6.7 |
| 7 channels + reverse retrieval | 0.982 | ≈67 |
| **after pruning (final)** | **0.975** | **6.5** |

(Measured on a held-out training sample; the oracle macro F0.5 of the Run 2 candidate set was ≈0.985.)

---

## 4. Matching Model

**Features used** (~100 per pair):
- **Name:** rapidfuzz ratio / token-sort / token-set / partial ratio on full and core names; Jaro-Winkler,
  ratio and partial ratio on compact names (catch `rkkproducts` vs `Rkk Products Pvt Ltd`); token-sort /
  token-set on phonetic skeletons (cross-script); token Jaccard, first-token agreement, shared-token
  count, token counts.
- **Address:** ratio / token-sort / token-set / partial ratio on normalized addresses and on their
  alphabetic part; empty-address flag (address features set to missing).
- **House numbers / postal-like numbers:** number-set Jaccard, containment both ways, first-number
  equality, digit edit distance, log absolute and relative difference of the primary number, closest
  number difference, street-name equality and Jaro-Winkler — separating typos (54 vs 54-56) from
  look-alike decoys (5116 vs 5121).
- **Rarity:** how many S1 / pool records of the same country share the exact core name, compact name and
  address (plain counts; see §5 for why not rates).
- **Competition (pool side):** for the candidate record, the best and second-best claim it receives from
  *any* S1 per channel → margin of this pair against the strongest other claimant (a soft, learnable
  version of the one-owner constraint); number of S1 lists containing it.
- **Context (S1 side):** gap to and rank against the S1's best candidate on name, address and combined
  similarity; number of candidates; retrieval scores of the joint and address channels; record source
  (S2/S3), native-script and web-name flags.
- **Country** is never a feature, so the model has no path that only works for US/India.

**Model type:** scikit-learn `HistGradientBoostingClassifier` (800 iterations, 63 leaves, L2 1.0,
learning rate 0.08), trained on **7.33 M pruned pairs from ~1.1 M training S1 entities** (samples B and C,
≈51 % of the training S1; the pruner is trained on a disjoint sample A).

**Training in a test-like world.** The test has 5.75 S2/S3 records per S1 vs 4.67 in train. Setting aside
18.8 % of the training S1 entities turns their S2/S3 records into unmatched look-alikes and reproduces the
test density (5.77); rarity, competition and labels are computed in that world.

**France (no labels).** France is scored by the same matcher; no French-specific model is used. We also
tried self-training (retraining with confident French pseudo-labels from the model's own scores and the
one-owner constraint); it made the model drop same-name, empty-address French matches that were in fact
mostly correct and lowered the leaderboard score (0.9500 vs 0.9533), so it is **not** used.

**Threshold selection method:** F0.5-optimized. Decisions: (1) exclusivity — each S2/S3 record is kept
only for the S1 with the highest probability (ground truth is a partition); (2) probability cutoff. On
grouped validation the macro F0.5 is flat for cutoffs 0.6–0.8 (0.9733 at 0.7), and the per-pair
precision needed for a match to raise F0.5 is ≈ F/1.25 ≈ 0.76. On the test set the model is less
precise than in validation, and French mid-range matches are claimed more strongly by another S1 far
more often (15–19 % vs <1.5 % for US/India, a label-free signal of look-alike confusion). Final cutoffs:
**0.85 for US / India, 0.95 for France** (0.85 for any other country).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** grouped validation in the test-like simulation **0.9733** (India 0.963, US 0.980;
  matcher trained on sample B); public leaderboard **0.963** (final).

Progression (validation on held-out S1 entities; experiment log `experiments/results.jsonl`):

| step | validation macro F0.5 | change |
|---|---|---|
| non-ML baseline: cutoff on joint retrieval score | 0.712 | |
| M1 gradient boosting, v1 features | 0.954 | +0.242 |
| M2 + rarity counts + house-number precision | 0.965 | +0.011 |
| M3 + pool-side competition margins | 0.965 (+0.002 in a like-for-like re-run) | |
| M4 + stage-2 entity-context stacking (not kept: +0.0006) | 0.966 | |
| Run 2: 7 channels, pruner, 3× training data, test-like simulation | 0.968 | India +0.004 |
| **Run 6: + reverse retrieval** | **0.973** | **+0.006** |

Leaderboard: 0.9426 (M3, 33 cands/S1, cutoff 0.7) → 0.944 (Run 2, 6.9 cands/S1) → 0.9527 (shift-robust
features + cutoff 0.8) → 0.9533 (per-country cutoffs) → 0.9538 (matcher trained on 1.7× more entities) → **0.963 (+ reverse
retrieval, final)**; France
self-training 0.9500 (rejected). A probe submission with French rows left empty
(0.817) showed France at ≈0.89 and US/India at ≈0.95 on the public split.

**Train/test shift fixes that mattered.** Name/address frequencies expressed as rates (count / world size)
shifted for every record because the test has half as many US S1 records — plain counts are identical
for the typical record; retrieval ranks and name-channel scores depend on how crowded a neighbourhood
is (more crowded in test), so the matcher does not see them and compares names through string features.

- **Common false positives (wrong merges):** look-alike decoys of a real entity on the same street with a
  nearby house number (5116 vs 5121 Gloria St; 2 vs 23 Rue Georges Clemenceau), often with a swapped
  legal form; exact-name records with an empty address assigned to the wrong one of several same-name
  entities; different businesses in the same building; in France, generic names that contain the city
  ("Nantes Pharmacie", "Saint-Nazaire Médico").
- **Common false negatives (missed matches):** records with an empty address and a name shared by other
  entities (unresolvable from the name alone, by design left unmatched under F0.5); native-script names
  with truncated addresses whose transliteration differs strongly from the English spelling; records whose
  name was replaced by an unrelated trade name and whose address is partial; entities with many matches
  where one copy has several simultaneous typos (lost at retrieval: ~1.8 % of true pairs, plus ~0.7 %
  removed by the pruner).

---

## 6. Conclusion

A retrieval-then-classify cascade with script-aware normalization, targeted retrieval channels and a
learned pruner reaches 0.975 pair recall at about 7 candidates per entity and 0.963 macro F0.5 on the public
leaderboard. The largest lessons: exact re-scoring matters in capped TF-IDF retrieval; retrieving in both directions
recovers true records that crowded forward lists push out (our single largest gain); features must be
invariant to the size and crowdedness of the dataset they are computed on, because the test differs from
training in both; and with an unseen country, label-free signals (the one-owner constraint, look-alike
conflicts) are the only safe guide for decisions. More labelled training entities gave a further, small
but real, gain.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/ber/` (all source), `configs/` (experiment configs),
`README.md` (exact commands, ~4 h end to end on 12 CPU threads), `requirements.txt` (pinned).
Entry points, in order: `ber.io`, `ber.normalize`, `ber.blocking` (per channel), `ber.cascade pruner /
trainset`, `ber.run5 trainset / final / test`, `ber.finalize run6`. They regenerate
`output/candidate_pairs.tsv` (pruned candidates = pairs scored by the matcher) and
`output/matching_results.tsv`.

### B. Additional Results

Blocking recall by country (held-out train sample, before pruning): 7 forward channels India 0.951, US 0.978;
with reverse retrieval India 0.968, US 0.992.
Per-entity F0.5 by number of true matches (M1 analysis): singletons 0.947, 1 match 0.884, 2: 0.942,
3: 0.957, 4: 0.963, 5+: 0.966 — single-match entities are the hardest (one miss costs the whole entity).
