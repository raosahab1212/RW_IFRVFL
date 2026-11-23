"""Compare gbtsvm.py with the authors' Python code. usage: python checks/compare_gbtsvm.py PATH_TO_REFERENCE_CODE"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(1, sys.argv[1])
os.chdir(os.path.join(HERE, ".."))
import numpy as np, datasets as D, evaluate as E
from sklearn.cluster import KMeans
import gen_ball, GBTSVM as ref
from gbtsvm import GBTSVMClassifier
gen_ball.KMeans = lambda n_clusters: KMeans(n_clusters=n_clusters, n_init=10, random_state=0)  # seed the reference's 2-means as ours
same_balls = same_pred = total = 0
for ds in ["heart", "pima", "wisconsin", "ionosphere", "australian"]:
    X, y, _ = D.load(ds)
    for f, (tr, te) in enumerate(E.outer_splits(ds, y)):
        if f > 1: break
        pre = D.make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
        A, B = np.asarray(pre.transform(X.iloc[tr]), float), np.asarray(pre.transform(X.iloc[te]), float)
        ya, yb = np.where(y[tr] == 1, 1.0, -1.0), np.where(y[te] == 1, 1.0, -1.0)
        for pur, mb in [(0.925, 4), (0.85, 1)]:
            balls = gen_ball.gen_balls(np.c_[A, ya], pur=pur, delbals=mb)
            Cr = np.array([b[0] for b in balls]); Rr = np.array([b[1] for b in balls]); Lr = np.array([b[2] for b in balls], float)
            for d in [1e-5, 0.1, 10.0]:
                clf = GBTSVMClassifier(d1=d, d2=d, purity=pur, min_ball_size=mb, random_state=0).fit(A, ya)
                from gbtsvm import granular_balls
                Co, Ro, Lo = granular_balls(A, ya, pur, 1, mb, 0)
                sb = Co.shape == Cr.shape and np.allclose(Co, Cr) and np.allclose(Ro, Rr) and np.array_equal(Lo, Lr)
                acc_ref, _ = ref.GBTSVM(np.c_[Cr, Rr, Lr], np.c_[B, yb], d, d)
                acc_our = 100 * np.mean(clf.predict(B) == yb)
                same_balls += sb; same_pred += abs(acc_ref - acc_our) < 1e-9; total += 1
print(f"GBTSVM: identical granular balls in {same_balls}/{total} cases; identical test accuracy in {same_pred}/{total} cases")
