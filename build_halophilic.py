"""Build data/halophilic.csv (Zhang and Ge, 2013) with the 442 composition features of Nath (2016)."""
import itertools
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "zhang139_src", "zhang139.fasta")
OUT = os.path.join(HERE, "data", "halophilic.csv")

AA = "ARNDCQEGHILKMFPSTWYV"
GROUPS = {
    "Tiny": "ACGST", "Small": "ANDCGPSTV", "Aliphatic": "ILV", "Non-polar": "ACGILMFPWYV",
    "Aromatic": "HFWY", "Polar": "RNDQEHKST", "Charged": "RDEHK", "Basic": "RHK",
    "Acidic": "DE", "Hydrophobic": "ACILMFWYV", "Hydrophilic": "NDQEK",
}
LABEL = {"Salinibacter ruber": "halophilic", "Chlorobium luteolum": "non-halophilic"}


def read_fasta(path):
    recs, head, seq = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith(">"):
            if head:
                recs.append((head, "".join(seq)))
            head, seq = line[1:], []
        elif line:
            seq.append(line.upper())
    if head:
        recs.append((head, "".join(seq)))
    return recs


def features(seq):
    n = max(len(seq), 1)
    pairs = [seq[i:i + 2] for i in range(len(seq) - 1)]
    m = max(len(pairs), 1)
    f = {a: 100 * seq.count(a) / n for a in AA}
    counts = pd.Series(pairs).value_counts() if pairs else pd.Series(dtype=int)
    f.update({a + b: 100 * counts.get(a + b, 0) / m for a, b in itertools.product(AA, AA)})
    for g, members in GROUPS.items():
        f[g] = 100 * sum(seq.count(a) for a in members) / n
    for g, members in GROUPS.items():
        f[g + "-2gram"] = 100 * sum(p[0] in members and p[1] in members for p in pairs) / m
    return f


def main():
    rows = []
    for head, seq in read_fasta(SRC):
        org = next((o for o in LABEL if f"OS={o}" in head), None)
        if org is None:
            raise ValueError(f"unknown organism in header: {head[:80]}")
        rows.append({**features(seq), "accession": head.split("|")[1], "class": LABEL[org]})
    df = pd.DataFrame(rows)
    assert len(df) == 278 and df["class"].value_counts().tolist() == [139, 139], df["class"].value_counts()
    assert df.shape[1] == 442 + 2
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {df.shape[0]} proteins, {df.shape[1] - 2} features, {df['class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
