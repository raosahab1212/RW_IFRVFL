"""
Experiment definitions and deterministic sharding.

A task is (experiment, dataset, model, label_noise, feat_noise, fold).
`shard_tasks(exp, n_shards, slots)` packs tasks into n_shards * slots worker
slots with longest-processing-time-first, using rough per-fold cost estimates
(seconds) measured in the Phase-0 smoke test and the Adult fold-0 timing run.
Every shard job gets the same assignment because the packing is deterministic.
"""

from datasets import BENCHMARK, CASE_STUDIES, LARGE
from models import ABLATION_MODELS, BENCHMARK_MODELS

FOLDS = range(10)
NOISE_DATASETS = ["australian", "wisconsin", "heart", "ionosphere", "magic"]
NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40]
_ABLATION_ONLY = [m for m in ABLATION_MODELS if m not in BENCHMARK_MODELS]

EXPERIMENTS = {
    "e1": dict(datasets=BENCHMARK, models=BENCHMARK_MODELS, label=[0.0], feat=[0.0]),
    "e7": dict(datasets=CASE_STUDIES, models=BENCHMARK_MODELS, label=[0.0], feat=[0.0]),
    "e3": dict(datasets=BENCHMARK, models=_ABLATION_ONLY, label=[0.0], feat=[0.0]),
    "e4_label": dict(datasets=NOISE_DATASETS, models=BENCHMARK_MODELS, label=NOISE_LEVELS, feat=[0.0]),
    "e4_feature": dict(datasets=NOISE_DATASETS, models=BENCHMARK_MODELS, label=[0.0], feat=NOISE_LEVELS),
}

# Rough seconds per outer fold with the full protocol (30 configs).
SMALL_COST = {"SVM": 15, "RF": 200, "XGBoost": 110, "LightGBM": 25, "CatBoost": 450,
              "NB": 5, "KNN": 6, "NF-RVFL": 12, "ANFIS": 350, "GBTSVM": 600,
              "TRKM": 15, "RVFL": 8, "IFRVFLC": 15, "RW-IFRVFL": 9}
LARGE_COST = {"SVM": 420, "RF": 520, "XGBoost": 190, "LightGBM": 95, "CatBoost": 400,
              "NB": 5, "KNN": 20, "NF-RVFL": 35, "ANFIS": 7200, "GBTSVM": 1020,
              "TRKM": 215, "RVFL": 30, "IFRVFLC": 400, "RW-IFRVFL": 60}
WIDE = {"protein": 3.0, "mushroom": 1.5, "adult": 1.0}   # many encoded features


def cost(t):
    _, ds, model, *_ = t
    base = LARGE_COST if ds in LARGE else SMALL_COST
    c = base.get(model, base["RW-IFRVFL"])          # ablation variants ~ RW-IFRVFL
    return c * WIDE.get(ds, 1.0)


def all_tasks(exp, datasets=None, models=None):
    e = EXPERIMENTS[exp]
    ds_list = [d for d in e["datasets"] if not datasets or d in datasets]
    md_list = [m for m in e["models"] if not models or m in models]
    return [(exp, d, m, ln, fn, f) for d in ds_list for m in md_list
            for ln in e["label"] for fn in e["feat"] for f in FOLDS]


def shard_tasks(exp, n_shards, slots=4, datasets=None, models=None):
    """Return list over shards; each shard is a list of `slots` task lists."""
    tasks = sorted(all_tasks(exp, datasets, models), key=lambda t: (-cost(t), t[1:]))
    bins = [[] for _ in range(n_shards * slots)]
    load = [0.0] * len(bins)
    for t in tasks:
        i = min(range(len(bins)), key=lambda b: (load[b], b))
        bins[i].append(t)
        load[i] += cost(t)
    return [bins[s * slots:(s + 1) * slots] for s in range(n_shards)], load


def result_name(t):
    exp, ds, model, ln, fn, f = t
    safe = model.replace(":", "_")
    return f"{exp}/{ds}__{safe}__L{ln:.2f}__F{fn:.2f}__f{f}.csv"


if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "e1"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    shards, load = shard_tasks(exp, n)
    total = sum(map(cost, all_tasks(exp)))
    print(f"{exp}: {len(all_tasks(exp))} tasks, ~{total / 3600:.0f} CPU-h estimated, "
          f"{n} shards x 4 slots, longest slot ~{max(load) / 3600:.1f} h")
