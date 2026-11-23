"""Experiment definitions and deterministic sharding of their tasks across jobs."""
from datasets import BENCHMARK, CASE_STUDIES, LARGE
from models import ABLATION_MODELS, COMPONENT_MODELS, MAIN_MODELS, SENS_MODELS

FOLDS = range(10)
NOISE_DATASETS = ["australian", "wisconsin", "heart", "ionosphere", "magic"]
NOISE_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.40]

EXPERIMENTS = {
    "benchmark": dict(datasets=BENCHMARK, models=MAIN_MODELS, label=[0.0], feat=[0.0]),
    "label_noise": dict(datasets=NOISE_DATASETS, models=MAIN_MODELS, label=NOISE_LEVELS, feat=[0.0]),
    "feature_noise": dict(datasets=NOISE_DATASETS, models=MAIN_MODELS, label=[0.0], feat=NOISE_LEVELS),
    "case_study": dict(datasets=CASE_STUDIES, models=MAIN_MODELS + COMPONENT_MODELS, label=[0.0], feat=[0.0]),
    "ablation": dict(datasets=BENCHMARK, models=ABLATION_MODELS, label=[0.0], feat=[0.0]),
    "ablation_label": dict(datasets=NOISE_DATASETS, models=ABLATION_MODELS, label=NOISE_LEVELS, feat=[0.0]),
    "ablation_feature": dict(datasets=NOISE_DATASETS, models=ABLATION_MODELS, label=[0.0], feat=NOISE_LEVELS),
    "sensitivity": dict(datasets=BENCHMARK, models=SENS_MODELS, label=[0.0], feat=[0.0]),
    "sensitivity_label": dict(datasets=NOISE_DATASETS, models=SENS_MODELS, label=[0.30], feat=[0.0]),
}

SMALL_COST = {"SVM": 15, "RF": 200, "XGBoost": 110, "LightGBM": 25, "CatBoost": 450,
              "NB": 5, "KNN": 6, "NF-RVFL": 12, "ANFIS": 350, "GBTSVM": 600,
              "TRKM": 15, "RVFL": 8, "IFRVFLC": 15, "RW-IFRVFL": 9}
LARGE_COST = {"SVM": 420, "RF": 520, "XGBoost": 190, "LightGBM": 95, "CatBoost": 400,
              "NB": 5, "KNN": 20, "NF-RVFL": 35, "ANFIS": 7200, "GBTSVM": 1020,
              "TRKM": 215, "RVFL": 30, "IFRVFLC": 400, "RW-IFRVFL": 60}
WIDE = {"mushroom": 1.5, "adult": 1.0}


def cost(t):
    _, ds, model, *_ = t
    if model.startswith("sens:"):
        return (25 if ds in LARGE else 1) * WIDE.get(ds, 1.0)
    base = LARGE_COST if ds in LARGE else SMALL_COST
    return base.get(model, base["RW-IFRVFL"]) * WIDE.get(ds, 1.0)


def all_tasks(exp, datasets=None, models=None):
    e = EXPERIMENTS[exp]
    ds_list = [d for d in e["datasets"] if not datasets or d in datasets]
    md_list = [m for m in e["models"] if not models or m in models]
    return [(exp, d, m, ln, fn, f) for d in ds_list for m in md_list
            for ln in e["label"] for fn in e["feat"] for f in FOLDS]


def shard_tasks(exp, n_shards, slots=4, datasets=None, models=None):
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
    return f"{exp}/{ds}__{model.replace(':', '_')}__L{ln:.2f}__F{fn:.2f}__f{f}.csv"


if __name__ == "__main__":
    for name in EXPERIMENTS:
        tasks = all_tasks(name)
        print(f"{name:18s} {len(tasks):5d} tasks, ~{sum(map(cost, tasks)) / 3600:.0f} CPU-h")
