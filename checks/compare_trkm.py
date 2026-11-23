import sys, os, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(1, sys.argv[1])
os.chdir(os.path.join(HERE, ".."))
import numpy as np, datasets as D, evaluate as E
import TRKM as ref
from trkm import TRKMClassifier
captured = {}
_acc = ref.accuracy_score
def cap(obs, pred):
    captured["pred"] = np.asarray(pred).ravel(); return _acc(obs, pred)
ref.accuracy_score = cap
agree = total = 0; worst = 1.0
for ds in ["heart", "pima", "wisconsin", "ionosphere", "australian"]:
    X, y, _ = D.load(ds)
    for f, (tr, te) in enumerate(E.outer_splits(ds, y)):
        if f > 2: break
        pre = D.make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
        A, B = np.asarray(pre.transform(X.iloc[tr]), float), np.asarray(pre.transform(X.iloc[te]), float)
        ya, yb = np.where(y[tr] == 1, 1.0, -1.0), np.where(y[te] == 1, 1.0, -1.0)
        for eta, lam, mew in [(0.1, 1e-3, 4.0), (1.0, 0.1, 0.5), (10, 1.0, 8.0)]:
            ref.TRKM(np.c_[A, ya], np.c_[B, yb], eta, eta, lam, lam, mew)
            p_ref = captured["pred"]
            p_our = TRKMClassifier(eta1=eta, eta2=eta, lambd1=lam, lambd2=lam, mew=mew).fit(A, ya).predict(B)
            a = float(np.mean(p_ref == np.asarray(p_our, float)))
            agree += int(a == 1.0); total += 1; worst = min(worst, a)
print(f"TRKM: identical test predictions in {agree}/{total} (dataset, fold, setting) cases; lowest agreement {worst:.4f}")
