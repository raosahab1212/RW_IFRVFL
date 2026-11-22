# RW-IFRVFL experiments

Experiment code for the RW-IFRVFL classifier (reliability-weighted intuitionistic
fuzzy RVFL) and the baselines it is compared with. All reported numbers are produced
by these scripts; tables are generated from the result CSVs.

## Layout

| File | Purpose |
|---|---|
| `datasets.py` | Loads the 14 benchmark datasets + protein case study; no preprocessing, no oversampling. `python datasets.py` prints the dataset summary table. |
| `rw_ifrvfl.py` | RW-IFRVFL implemented step by step from the manuscript (Algorithm 1; spread from Sec. 4.4; search space from Table 5). Used for RW-IFRVFL in every experiment. |
| `rvfl_family.py` | One RVFL estimator (`FuzzyRVFL`) covering RVFL, IFRVFLC and all ablation variants (same random layer and solver; its `rw` variant equals `rw_ifrvfl.py`). IFRVFLC follows Mishra, Gupta & Hazarika (2023), Eqs. 22–24, with their kernel-width and radius grids. |
| `gbtsvm.py` | GBTSVM, checked against the authors' reference code. |
| `trkm.py` | TRKM, checked against the authors' reference code. |
| `models.py` | Model registry, search spaces, equal tuning budget (30 configs). |
| `evaluate.py` | Protocol: saved outer 10-fold splits, nested tuning inside the training folds, train-only noise, metrics and timings. |
| `smoke_test.py` | Quick check: all models, 2 folds, small budget. |
| `legacy/` | Earlier implementations; ANFIS and NF-RVFL are used from here. |
| `splits/` | Outer split indices, created once per dataset and reused everywhere. |

## Protocol

- Outer: stratified 10-fold CV, seed 42; the indices are saved in `splits/<dataset>.npz`.
- Inner: 5-fold CV on the outer-training part (3-fold on a stratified 5,000-row
  subsample when N > 5,000); criterion is macro-F1; up to 30 configurations per model.
- Preprocessing is fitted on the training fold only: median / most-frequent
  imputation, one-hot encoding, then min-max scaling.
- Positive class = minority class. Reported: accuracy, macro-F1, minority-F1,
  AUC (from continuous scores), fit time and predict time (single-threaded).
- Label noise is flipped on training labels only, using the same splits; models
  are retuned at each level. Test labels are never corrupted.
- O(N^2) models (SVM, TRKM, ANFIS, GBTSVM) train on a capped stratified subsample
  on large datasets (`models.CAPPED`) (reported with the results).

## Running on GitHub Actions

Actions tab -> `experiments` -> Run workflow, choosing the experiment
(`e1` benchmark, `e3` ablation, `e4_label` / `e4_feature` noise, `e7` case study)
and the number of shards. Each shard job uploads its per-task CSVs; the `merge`
job combines them into `results-<experiment>` (`<exp>_all.csv`, `<exp>_summary.csv`).
Re-running a failed job only recomputes tasks whose result files are missing
within that job.

Locally: `python run_shard.py e1 0 1 --datasets heart --models RW-IFRVFL`, then
`python merge.py`.

GBTSVM and TRKM follow the original authors' public implementations.
