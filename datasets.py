"""
Dataset loading for the RW-IFRVFL experiments.

`load(name)` returns (X, y, info):
    X     pandas DataFrame, raw features, missing values as NaN
    y     int array in {0, 1}; 1 = minority class (positive class for
          minority-F1 and for AUC); ties broken by sorted label order
    info  dict(name, n, d_raw, categorical columns, label names, imbalance ratio)

Nothing is imputed, encoded or scaled here: that happens inside each training
fold via `make_preprocessor` so no information leaks from the test fold.
No oversampling is applied anywhere.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BENCHMARK = [
    "adult", "australian", "breast", "heart", "horse-colic", "ionosphere",
    "magic", "mammographic", "monk-2", "mushroom", "phoneme", "pima",
    "wisconsin", "diabetes",
]
CASE_STUDIES = ["halophilic"]
LARGE = {"adult", "magic", "mushroom", "phoneme", "diabetes", "protein"}


def _read_keel(path):
    names, nominal = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.lower().startswith("@attribute"):
                m = re.match(r"@attribute\s+('[^']+'|\S+)\s+(.*)", line.strip(), re.I)
                names.append(m.group(1).strip("'"))
                nominal.append("{" in m.group(2))
            elif line.lower().startswith("@data"):
                break
    df = pd.read_csv(path, comment="@", header=None, names=names,
                     skipinitialspace=True, na_values=["?"], dtype=str)
    target = names[-1]
    cat = [c for c, nom in zip(names[:-1], nominal[:-1]) if nom]
    X = df.drop(columns=target)
    for c in X.columns:
        if c not in cat:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, df[target].str.strip(), cat


def _raw(name):
    p = lambda f: os.path.join(DATA_DIR, f)
    if name == "wisconsin":
        df = pd.read_csv(p("wisonsin.csv")).drop(columns=["id", "Unnamed: 32"], errors="ignore")
        return df.drop(columns="diagnosis"), df["diagnosis"], []
    if name == "ionosphere":
        df = pd.read_csv(p("ionosphere.data"), header=None)
        return df.iloc[:, :-1], df.iloc[:, -1], []
    if name == "diabetes":
        df = pd.read_csv(p("diabetes_binary_5050split_health_indicators_BRFSS2015.csv"))
        return df.drop(columns="Diabetes_binary"), df["Diabetes_binary"].astype(int).astype(str), []
    if name == "halophilic":
        df = pd.read_csv(p("halophilic.csv"))
        return df.drop(columns=["class", "accession"]), df["class"], []
    if name == "protein":
        df = pd.read_csv(p("protien.csv"))
        return df.drop(columns="class"), df["class"], []
    return _read_keel(p(f"{name}.dat"))


def load(name):
    X, labels, cat = _raw(name)
    labels = labels.astype(str).str.strip()
    counts = labels.value_counts()
    if len(counts) != 2:
        raise ValueError(f"{name}: expected 2 classes, got {counts.to_dict()}")
    ordered = sorted(counts.index, key=lambda l: (counts[l], l))   # minority first
    pos = ordered[0]
    y = (labels == pos).to_numpy().astype(int)
    X = X.reset_index(drop=True)
    X.columns = [str(c) for c in X.columns]
    cat = [str(c) for c in cat]
    # drop constant columns (no information, breaks some scalers)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    cat = [c for c in cat if c in X.columns]
    info = dict(
        name=name, n=len(y), d_raw=X.shape[1], categorical=cat,
        positive=pos, negative=ordered[1],
        n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
        imbalance=round(counts.max() / counts.min(), 2),
        missing=int(X.isna().sum().sum()),
    )
    return X, y, info


def make_preprocessor(X, scale=True):
    """Impute + one-hot + min-max, fitted on the training fold only."""
    cat = [c for c in X.columns if X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    steps = []
    if num:
        steps.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        steps.append(("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat))
    ct = ColumnTransformer(steps)
    return make_pipeline(ct, MinMaxScaler()) if scale else ct


if __name__ == "__main__":
    rows = []
    for n in BENCHMARK + CASE_STUDIES:
        X, y, info = load(n)
        d_enc = make_preprocessor(X).fit_transform(X).shape[1]
        rows.append({**{k: v for k, v in info.items() if k != "categorical"},
                     "n_categorical": len(info["categorical"]), "d_encoded": d_enc})
    print(pd.DataFrame(rows).to_string(index=False))
