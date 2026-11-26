"""Run one shard of an experiment: python run.py <experiment> <shard> <n_shards> [--slots 4]."""
import argparse
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
HW = dict(host_cpu=platform.processor() or platform.machine(), host_cores=os.cpu_count(),
          host_os=platform.platform(), python=platform.python_version())


def run_slot(task_list):
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:  # only ANFIS needs torch
        pass
    import datasets as D
    import evaluate as E
    from experiments import EXPERIMENTS, result_name

    cache, log = {}, []
    for t in task_list:
        exp, ds, model, ln, fn, fold = t
        path = os.path.join(OUT, result_name(t))
        if os.path.exists(path):
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if ds not in cache:
            cache = {ds: D.load(ds)}
        X, y, _ = cache[ds]
        t0 = time.perf_counter()
        try:
            cfg = EXPERIMENTS[exp]
            rows = E.run_outer(model, X, y, ds, label_noise=ln, feat_noise=fn, folds=[fold],
                               label_kind=cfg.get("label_kind", "random"), evasion=cfg.get("evasion", False))
            for r in rows:
                r.update(experiment=exp, wall_s=time.perf_counter() - t0, **HW)
            pd.DataFrame(rows).to_csv(path, index=False)
            msg = f"ok   {result_name(t)} {time.perf_counter() - t0:.0f}s acc={rows[0]['accuracy']:.4f}"
        except Exception as e:
            msg = f"FAIL {result_name(t)} {e!r}\n{traceback.format_exc()}"
        print(msg, flush=True)
        log.append(msg)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("shard", type=int)
    ap.add_argument("n_shards", type=int)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--models", default="")
    a = ap.parse_args()

    from experiments import shard_tasks
    split = lambda s: [x for x in s.split(",") if x] or None
    shards, _ = shard_tasks(a.experiment, a.n_shards, a.slots, split(a.datasets), split(a.models))
    slots = [s for s in shards[a.shard] if s]
    print(f"shard {a.shard}/{a.n_shards}: {sum(map(len, slots))} tasks in {len(slots)} slots", flush=True)
    if not slots:
        return
    with ProcessPoolExecutor(len(slots)) as ex:
        logs = list(ex.map(run_slot, slots))
    fails = [m for lg in logs for m in lg if m.startswith("FAIL")]
    print(f"done: {sum(map(len, logs)) - len(fails)} ok, {len(fails)} failed", flush=True)


if __name__ == "__main__":
    main()
