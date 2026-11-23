"""Compare nfrvfl.py with the authors' MATLAB code. See checks/README.md.

usage: python checks/compare_nfrvfl.py DIR   (DIR holds case1.mat and case2.mat written by nfrvfl_reference_run.m)
"""
import os, sys, numpy as np, scipy.io as sio
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import nfrvfl
DIR = sys.argv[1]

class Identity:
    def fit(self, X): return self
    def transform(self, X): return np.asarray(X, float)

for c in (1, 2):
    m = sio.loadmat(os.path.join(DIR, f"case{c}.mat"))
    mats = iter([m["Alpha"], m["W"]])
    class Rng:
        def uniform(self, lo, hi, size): return next(mats)
    class FixedKMeans:
        def __init__(self, **k): pass
        def fit(self, X): self.cluster_centers_ = m["centres"]; return self
    nfrvfl.StandardScaler, nfrvfl.KMeans = Identity, FixedKMeans
    nfrvfl.np.random.default_rng = lambda seed: Rng()
    N, L, C = int(m["N"].item()), int(m["L"].item()), float(m["C"].item())
    clf = nfrvfl.NeuroFuzzyRVFLClassifier(n_hidden=L, reg=1.0 / C, n_rules=N)
    clf.fit(m["trX"], m["trY"].ravel())
    s_py = clf.decision_function(m["teX"])
    s_ml = m["scores"]
    acc_py = 100 * np.mean(clf.predict(m["teX"]) == m["teY"].ravel())
    rel = np.abs(s_py - s_ml).max() / np.abs(s_ml).max()
    print(f"case {c}: N={N} L={L} C={C}: max rel. difference of test scores {rel:.2e}; "
          f"test accuracy MATLAB code (Octave) {float(m['acc'].item()):.4f}, Python port {acc_py:.4f}; "
          f"identical predictions: {bool(np.all(np.argmax(s_py,1) == np.argmax(s_ml,1)))}")
