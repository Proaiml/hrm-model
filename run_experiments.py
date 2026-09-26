"""
HRM deneyleri: aynı veri, aynı güncelleme bütçesi, birden çok tohum.

    py -3.11 run_experiments.py --task maze --methods all --seeds 0 1 2
    py -3.11 run_experiments.py --task direction --methods pos_ablation

Sonuçlar: results/<görev>/<yöntem>_s<tohum>.json   (make_report.py tablo ve grafikleri üretir)
Model ağırlıkları: checkpoints/<görev>_<yöntem>_s<tohum>.pt  (demo.py kullanır)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hrm_lab import models as M  # noqa: E402
from hrm_lab import tasks as T   # noqa: E402
from hrm_lab import train as TR  # noqa: E402

TASKS = {
    "direction": dict(spec=T.DIRECTION, train=2000, test=1000, steps=1500),
    "maze":      dict(spec=T.maze_spec(9), train=1000, test=1000, steps=5000),
    "sudoku":    dict(spec=T.SUDOKU, train=1000, test=1000, steps=5000),
}

# yöntem -> (model kurucu, eğitim ayarları)
METHODS = {
    "transformer2":   (lambda s: M.TransformerNet(s, layers=2), dict()),
    "transformer8":   (lambda s: M.TransformerNet(s, layers=8), dict()),
    "tinyhrm_bptt":   (lambda s: M.HRMReasoner(s, grad="bptt"), dict()),
    "hrm_onestep":    (lambda s: M.HRMReasoner(s, grad="one_step"), dict()),
    "hrm_ds":         (lambda s: M.HRMReasoner(s, grad="one_step"), dict(segments=4, eval_segments=16)),
    "hrm_ds_act":     (lambda s: M.HRMReasoner(s, grad="one_step"), dict(segments=4, eval_segments=16, act=True)),
    "looped_ds":      (lambda s: M.LoopedNet(s, layers=2, loops=4), dict(segments=4, eval_segments=16)),
    # konum ablasyonu (yön görevi): kullanıcının TinyHrm forward'ı, konumlu / konumsuz
    "tinyhrm_nopos":  (lambda s: M.HRMReasoner(s, grad="bptt", use_position=False), dict()),
}
GROUPS = {
    "all": ["transformer2", "transformer8", "tinyhrm_bptt", "hrm_onestep", "hrm_ds", "hrm_ds_act", "looped_ds"],
    "pos_ablation": ["tinyhrm_nopos", "tinyhrm_bptt"],
}


def dataset(task):
    """Aynı görev için tüm yöntemler aynı veriyi görür (önbellek: results/data_<görev>.npz)."""
    info = TASKS[task]
    path = ROOT / "results" / f"data_{task}.npz"
    if path.exists():
        z = np.load(path)
        keys = sorted(k for k in z.files)
        tr = tuple(z[k] for k in keys if k.startswith("train"))
        te = tuple(z[k] for k in keys if k.startswith("test"))
        return {"train": tr, "test": te}
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if task == "direction":
        tr, te = T.make_direction(info["train"], 1), T.make_direction(info["test"], 2)
    elif task == "maze":
        tr, te = T.make_maze(info["train"], 9, 1), T.make_maze(info["test"], 9, 2)
    else:
        tr, te = T.make_sudoku(info["train"], 1), T.make_sudoku(info["test"], 2)
    # eğitim ve test kümeleri kesişmesin
    seen = {x.tobytes() for x in tr[0]}
    keep = np.array([x.tobytes() not in seen for x in te[0]])
    te = tuple(a[keep] for a in te)
    np.savez_compressed(path, **{f"train{i}": a for i, a in enumerate(tr)}, **{f"test{i}": a for i, a in enumerate(te)})
    print(f"veri üretildi: {task} eğitim {len(tr[0])} test {len(te[0])} ({time.time() - t0:.0f} s)", flush=True)
    return {"train": tr, "test": te}


def run(task, method, seed, steps=None, out_dir=None):
    info = TASKS[task]
    spec = info["spec"]
    data = dataset(task)
    build, kw = METHODS[method]
    torch.manual_seed(seed)
    model = build(spec)
    cfg = TR.TrainConfig(steps=steps or info["steps"], seed=seed, augment=task != "direction", **kw)
    print(f"[{task} / {method} / tohum {seed}] parametre {M.count_params(model):,}", flush=True)
    fit = TR.train(model, spec, data, cfg, log_every=max(1, cfg.steps // 10))
    res, _ = TR.evaluate(model, spec, data["test"], cfg)
    out_dir = Path(out_dir or ROOT / "results" / task)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {"task": task, "method": method, "seed": seed, "params": M.count_params(model),
           "config": cfg, "train": fit, "test": res, "n_test": int(len(data["test"][0]))}
    TR.save_json(out_dir / f"{method}_s{seed}.json", row)
    ck = ROOT / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save({"method": method, "task": task, "state_dict": model.state_dict()}, ck / f"{task}_{method}_s{seed}.pt")
    print(f"[{task} / {method} / tohum {seed}] test tam doğruluk {res['exact']:.3f}  "
          f"(segmentlere göre {res['exact_by_segment'][:1]}...{res['exact_by_segment'][-1:]})  "
          f"{fit['train_seconds']} s, tepe bellek {fit['peak_mem_mb']} MB", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--methods", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    methods = [m for g in args.methods for m in GROUPS.get(g, [g])]
    for seed in args.seeds:
        for method in methods:
            if args.skip_done and (ROOT / "results" / args.task / f"{method}_s{seed}.json").exists():
                continue
            run(args.task, method, seed, args.steps)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
