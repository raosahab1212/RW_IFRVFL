# RW-IFRVFL: A Reliability-Weighted Intuitionistic Fuzzy Random Vector Functional-Link Classifier for Robust Binary Classification

Please cite the following paper if you are using this code.

Reference: P. Yadav, K. Khan, A. Shakya and A. K. Tiwari, "A Reliability-Weighted Intuitionistic Fuzzy Random Vector Functional-Link Classifier for Robust Binary Classification," Computers & Electrical Engineering (under review).

## Method

Each training sample receives an intuitionistic fuzzy pair: membership μ = (1 − f)·g and non-membership ν = f, where g is the Gaussian similarity of the sample to its class centroid and f is the fraction of its k nearest neighbours with a different label. The fused score S = (1 − α) + α(μ − ν) gives the sample weight |S| and the training target sign(S)·y of a weighted RVFL solved in closed form, β = (ZᵀDZ + λI)⁻¹ZᵀDỹ. α = 0 is the plain RVFL and α = 1/2 is the earlier score weighting (1 + μ − ν)/2.

## Experimental setup

The experiments were run on GitHub-hosted runners (Ubuntu 26.04, x86-64, 4 vCPUs) with Python 3.13.18 and the package versions in `requirements.txt`; every model fit used one thread. The processor varied between jobs: AMD EPYC 9V45, 9V74 or 7763, or Intel Xeon 6975P-C, 6973P-C, Platinum 8573C or Platinum 8370C (from the `lscpu` output in the job logs). The exception is `sensitivity_case` (280 tasks, accuracy only), which was run on a 2-vCPU Linux virtual machine (Ubuntu 24.04) with the same Python and package versions; the host of every task is recorded in the `host_*` columns of the result files. Only the accuracies of that experiment are used (`tab_sens_case.tex`), not its timings.

- Outer loop: stratified 10-fold cross-validation (seed 42); the fold indices are in `splits/`.
- Inner loop: 5-fold cross-validation on the training part (3-fold on a stratified 5,000-row subsample when N > 5,000), macro-F1 criterion, at most 30 configurations per model.
- Imputation, one-hot encoding and min-max scaling are fitted on the training part only.
- Label and feature noise are applied to the training part only, and every model is retuned at each noise level.
- Adversarial perturbations (`adversarial.py`) are built once per outer fold from surrogates fitted to the clean training part and shared by all models. Label attacks flip 5–40% of the training labels of each class, either the samples a logistic-regression surrogate classifies most confidently (`adv_label_far`) or those closest to its boundary (`adv_label_near`), following the farthest-first and nearest-first attacks of Xiao et al. (2015). The evasion attack (`adv_evasion`) moves each test sample by 20-step L∞ PGD on an MLP surrogate (two hidden layers of 64 ReLU units) with a per-feature budget of ε times the training standard deviation, ε ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1}, and reports a random-sign perturbation of the same budget as a control; models are trained on clean data, and categorical features are not perturbed. `adv_evasion_cont` repeats it with the integer-coded numeric columns (at most 10 distinct training values) also left unperturbed.

## Hyperparameters of RW-IFRVFL

| Parameter | Values |
|---|---|
| Hidden neurons L | 50, 100, 150, 200, 300 |
| Regularisation λ | 1e-4, 1e-3, 1e-2, 1e-1, 1 |
| Neighbours k | 3, 9, 25, 50 |
| Spread scale s | 0.5, 1, 1.5, 2 |
| Trust α | 0, 0.5, 0.75, 1 |

## Usage

```
pip install -r requirements.txt
python run.py benchmark 0 1 --slots 4 --datasets heart --models RW-IFRVFL
python merge.py
python make_tables.py tables
python make_figures.py figures
```

`run.py` skips every task whose result file already exists, and all result files are included in the repository, so on a fresh clone the command above runs nothing. To recompute tasks, move the corresponding files out of `results/<experiment>/` first, e.g. `mv results/benchmark results/benchmark_published`; `python merge.py` then reports any missing tasks.

Experiments: `benchmark`, `label_noise`, `feature_noise`, `case_study`, `ablation`, `ablation_label`, `ablation_feature`, `sensitivity`, `sensitivity_label`, `sensitivity_case`, `adv_label_far`, `adv_label_near`, `adv_evasion`, `adv_evasion_cont`. `python concentration.py` computes the distance-concentration and relabelling statistics of the case study (`results/distance_concentration.csv`, `results/relabel_halophilic.csv`). On GitHub: Actions → experiments → Run workflow; the merge job uploads `results-<experiment>`.

## Description of files

- `rw_ifrvfl.py`: RW-IFRVFL classifier.
- `rvfl_variants.py`: RVFL, IFRVFLC and the weighting variants of the ablation study.
- `gbtsvm.py`, `trkm.py`, `anfis.py`, `nfrvfl.py`: GBTSVM, TRKM, ANFIS and NF-RVFL baselines.
- `models.py`: model registry and hyperparameter grids.
- `datasets.py`: dataset loading and preprocessing.
- `evaluate.py`: nested cross-validation, noise injection and metrics.
- `adversarial.py`: surrogate-based adversarial label flips and PGD evasion sets (`adv/`, built by the workflow's prepare job).
- `experiments.py`: experiment definitions and sharding.
- `run.py`: runs one shard of an experiment.
- `merge.py`: combines the per-task files into `results/<experiment>_all.csv`.
- `make_tables.py`, `make_figures.py`: tables and figures of the paper.
- `concentration.py`: distance concentration of the centroid membership and share of relabelled samples.
- `checks/`: comparisons of `nfrvfl.py`, `trkm.py` and `gbtsvm.py` with the original authors' code.
- `build_halophilic.py`: builds `data/halophilic.csv` from the Zhang and Ge (2013) sequences.
- `data/`: the 14 benchmark datasets and the halophilic protein dataset (sources below).
- `results/`: per-task result files (`results/<experiment>/`) and the merged `results/<experiment>_all.csv` from which every table and figure is generated.
- `splits/`: outer-fold indices.

## Data sources

- Adult, Australian, Breast, Heart, Horse-Colic, Ionosphere, Magic, Mammographic, Monk-2, Mushroom, Phoneme, Pima and Wisconsin: UCI Machine Learning Repository (https://archive.ics.uci.edu) and KEEL repository (https://sci2s.ugr.es/keel/datasets.php).
- Diabetes: the balanced 50/50 split (70,692 rows, `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`) of the Diabetes Health Indicators dataset built from the CDC BRFSS 2015 survey (https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset); the full 253,680-row version is UCI dataset 891 (https://doi.org/10.24432/C53919).
- Halophilic: built by `build_halophilic.py` from the 278 sequences of Zhang and Ge (2013) in `data/zhang139_src/zhang139.fasta`, redistributed with its MIT licence (`data/zhang139_src/LICENSE_haloclass`).

## Remarks

NF-RVFL is a port of the public MATLAB code of Sajid, Malik, Tanveer and Suganthan (2024); GBTSVM and TRKM follow the public implementations of Quadir, Sajid and Tanveer (GBTSVM) and Quadir and Tanveer (TRKM); all three were checked against that code (`checks/`). The NF-RVFL port uses k-means rule centres (option 1 of the authors' code; their demo script selects fuzzy c-means) and the tansig activation of their demo, and its grid (`models.py`) is L ∈ {50, …, 300}, 5 to 50 rules and regularisation 1/C ∈ {1e-4, …, 1}. IFRVFLC follows Mishra, Gupta and Hazarika (2023).
