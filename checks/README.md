# Checks against the original authors' code

**NF-RVFL** (github.com/mtanveer1/NeuroFuzzy-RVFL, MATLAB). Copy `Neuro_Fuzzy_RVFL_train.m` to
`nf_train_given_centres.m`, replace the clustering block by `center = center_in;`, add `center_in` as the
last input and `PredictedTestLabel` as the last output (nothing else changes), and put it with
`credit_approval.mat`, `tansig.m` and `nfrvfl_reference_run.m` in one folder. Then

    octave-cli nfrvfl_reference_run.m          # writes case1.mat, case2.mat
    python checks/compare_nfrvfl.py FOLDER

Result (GNU Octave 8.4): maximum relative difference of the test scores 1.4e-15 (primal solve) and
3.0e-13 (dual solve); identical predictions and test accuracy (82.0144% and 89.9281%).

**TRKM** (github.com/mtanveer1/TRKM, `Classification.zip`) and **GBTSVM** (github.com/mtanveer1/GBTSVM,
`GBTSVM.zip`, with its 2-means seeded as in `gbtsvm.py`):

    python checks/compare_trkm.py PATH/Classification
    python checks/compare_gbtsvm.py PATH/GBTSVM

Result: TRKM identical test predictions in 45/45 cases; GBTSVM identical granular balls and test accuracy
in 60/60 cases (five datasets, saved outer folds, several hyperparameter settings).
