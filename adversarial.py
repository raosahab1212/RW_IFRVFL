"""Adversarial perturbation sets, built once per (dataset, outer fold) and shared by every model.

Label flips (training part): a logistic-regression surrogate is fitted to the clean training part;
within each class, "far" flips the most confidently correct samples first and "near" the samples
closest to the surrogate's boundary first (the farthest-first and nearest-first label attacks of
Xiao et al., Neurocomputing 160 (2015) 53-62).

Evasion (test part): an MLP surrogate is fitted to the clean training part and each test sample is
moved by 20-step L-infinity PGD on the surrogate's cross-entropy, with the per-feature budget
eps * (training standard deviation of the feature), as for the Gaussian feature noise. A random-sign
perturbation of the same budget is the control. Categorical (one-hot) features are not perturbed.

python adversarial.py [--datasets a,b]   writes adv/<dataset>__f<fold>.npz and adv/surrogates.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ADV_DIR = os.path.join(HERE, "adv")
SEED = 42
EVASION_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
PGD_STEPS = 20


def path(dataset, fold):
    return os.path.join(ADV_DIR, f"{dataset}__f{fold}.npz")


def load(dataset, fold):
    p = path(dataset, fold)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run adversarial.py first")
    return dict(np.load(p, allow_pickle=False))


def flipped_labels(y, tr, rate, kind, adv):
    """Training labels y[tr] with round(rate * n_c) labels of each class c flipped in surrogate order."""
    ytr = y[tr].copy()
    pos = {g: i for i, g in enumerate(tr)}
    for c in (0, 1):
        order = adv[f"{kind}_{c}"]
        n = int(round(rate * len(order)))
        ytr[[pos[g] for g in order[:n]]] = 1 - c
    return ytr


def evasion_tests(X, te, adv):
    """{(attack, eps): perturbed copy of X.iloc[te]} for the PGD and random-sign perturbations."""
    if not np.array_equal(adv["test_idx"], te):
        raise ValueError("adversarial set does not match the outer test fold")
    cols = [str(c) for c in adv["num_cols"]]
    out = {}
    for attack in ("pgd", "random"):
        for eps in EVASION_LEVELS:
            Xa = X.iloc[te].copy()
            Xa[cols] = Xa[cols].astype(float)
            Xa.loc[:, cols] = adv[f"{attack}_{eps:.2f}"]
            out[(attack, eps)] = Xa
    return out


def _mlp(d, seed):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))


def _train_mlp(Z, y, seed, epochs=60, batch=128):
    import torch
    net = _mlp(Z.shape[1], seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Zt, yt = torch.tensor(Z, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    loss = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for b in torch.randperm(len(Zt), generator=g).split(batch):
            opt.zero_grad()
            loss(net(Zt[b]).squeeze(1), yt[b]).backward()
            opt.step()
    return net.eval()


def _pgd(net, Z, y, eps_vec, steps=PGD_STEPS):
    import torch
    Z0 = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    e = torch.tensor(eps_vec, dtype=torch.float32)
    step = e * 2.5 / steps
    loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
    Za = Z0.clone()
    for _ in range(steps):
        Za.requires_grad_(True)
        g, = torch.autograd.grad(loss(net(Za).squeeze(1), yt), Za)
        with torch.no_grad():
            Za = torch.max(torch.min(Za + step * g.sign(), Z0 + e), Z0 - e)
    return Za.detach().numpy().astype(float)


def _acc(net, Z, y):
    import torch
    with torch.no_grad():
        return float(((net(torch.tensor(Z, dtype=torch.float32)).squeeze(1) > 0).numpy() == y).mean())


def build(dataset, fold, X, y, tr, te):
    from datasets import make_preprocessor
    pre = make_preprocessor(X.iloc[tr]).fit(X.iloc[tr])
    A, B = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
    num = [c for c in X.columns if X[c].dtype != object]
    ct, scaler = pre.steps[0][1], pre.steps[1][1]
    imputed = ct.transformers_[0][1].transform(X[num]) if num else np.zeros((len(X), 0))
    scale = scaler.scale_[:len(num)]                     # scaled = raw * scale + min
    sd = A.std(axis=0)
    unit = np.zeros(A.shape[1])
    unit[:len(num)] = sd[:len(num)]                      # one-hot columns get no budget

    ytr, yte = y[tr], y[te]
    out = dict(test_idx=te, num_cols=np.array(num, dtype=str))

    lr = LogisticRegression(C=1.0, max_iter=5000).fit(A, ytr)
    f = lr.decision_function(A)
    margin = (2 * ytr - 1) * f
    for c in (0, 1):
        idx = np.where(ytr == c)[0]
        out[f"far_{c}"] = tr[idx[np.lexsort((idx, -margin[idx]))]]
        out[f"near_{c}"] = tr[idx[np.lexsort((idx, np.abs(f[idx])))]]

    seed = SEED + fold
    net = _train_mlp(A, ytr, seed)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=B.shape)
    stats = dict(dataset=dataset, fold=fold, lr_train_acc=lr.score(A, ytr), lr_test_acc=lr.score(B, yte),
                 mlp_train_acc=_acc(net, A, ytr), mlp_test_acc=_acc(net, B, yte))
    raw_te = imputed[te]
    safe = np.where(scale > 0, scale, 1.0)
    for eps in EVASION_LEVELS:
        e = eps * unit
        Bp = _pgd(net, B, yte, e)
        stats[f"mlp_pgd_acc_{eps:.2f}"] = _acc(net, Bp, yte)
        for attack, Bx in (("pgd", Bp), ("random", B + signs * e)):
            delta = (Bx - B)[:, :len(num)] / safe          # back to raw feature units
            out[f"{attack}_{eps:.2f}"] = raw_te + delta
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="")
    a = ap.parse_args()
    import datasets as D
    from evaluate import outer_splits
    from experiments import NOISE_DATASETS
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    os.makedirs(ADV_DIR, exist_ok=True)
    chosen = [d for d in a.datasets.split(",") if d] or NOISE_DATASETS
    rows = []
    for ds in chosen:
        X, y, _ = D.load(ds)
        for fold, (tr, te) in enumerate(outer_splits(ds, y)):
            out, stats = build(ds, fold, X, y, tr, te)
            np.savez_compressed(path(ds, fold), **out)
            rows.append(stats)
            print(f"{ds} fold {fold}: MLP test acc {stats['mlp_test_acc']:.3f}, "
                  f"under PGD eps=0.4 {stats['mlp_pgd_acc_0.40']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ADV_DIR, "surrogates.csv"), index=False)


if __name__ == "__main__":
    main()
