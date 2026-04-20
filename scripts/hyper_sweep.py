"""
Hyperparameter sweep for Poles2025 snow pole detection.

Trains short runs (35 epochs) across a grid of:
  - learning rate : [0.001, 0.005, 0.01]
  - weight decay  : [0.0005, 0.001]
  - image size    : [640, 960]
  - optimizer     : [AdamW, SGD]

Results (val mAP@50, mAP@50-95, precision, recall) are appended to
  results/hyper_sweep_summary.csv
after each run so partial results survive early termination.

Usage:
    python scripts/hyper_sweep.py
    python scripts/hyper_sweep.py --data config/data_roadpoles_v1_cybele.yaml
    python scripts/hyper_sweep.py --epochs 40 --model yolo26s

The sweep uses augmentation settings from config/augmentation.yaml as the
base, overriding only the swept hyperparameters.  The goal is to identify
the best (lr, weight_decay, imgsz, optimizer) before the full model
comparison in compare_models.py.
"""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"
SWEEP_CSV = RESULTS_DIR / "hyper_sweep_summary.csv"

# Sweep grid — 3×2×2×2 = 24 configurations.
# At 35 epochs each on a single A100 this takes roughly 3–5 h total.
SWEEP_GRID = {
    "lr0":          [0.001, 0.005, 0.01],
    "weight_decay": [0.0005, 0.001],
    "imgsz":        [640, 960],
    "optimizer":    ["AdamW", "SGD"],
}

FIXED_AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.5,
    "degrees": 5.0,
    "translate": 0.15,
    "scale": 0.6,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "close_mosaic": 10,   # shorter run → close earlier
    "mixup": 0.05,
    "copy_paste": 0.1,
    "erasing": 0.4,
}

CSV_FIELDNAMES = [
    "run_name", "model", "lr0", "weight_decay", "imgsz", "optimizer",
    "epochs", "batch", "cos_lr",
    "val_precision", "val_recall", "val_mAP50", "val_mAP50_95",
    "train_seconds", "timestamp",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def append_csv(row: dict):
    """Append one result row; write header if file is new."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SWEEP_CSV.exists()
    with SWEEP_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_best_val_metrics(run_dir: Path) -> dict:
    """Read best epoch metrics from Ultralytics results.csv."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}

    def _f(row, key):
        try:
            return float(row.get(key, ""))
        except (ValueError, TypeError):
            return -1.0

    best = max(rows, key=lambda r: _f(r, "fitness"))
    return {
        "val_precision": _f(best, "metrics/precision(B)"),
        "val_recall":    _f(best, "metrics/recall(B)"),
        "val_mAP50":     _f(best, "metrics/mAP50(B)"),
        "val_mAP50_95":  _f(best, "metrics/mAP50-95(B)"),
    }


def run_single(
    model: str,
    data_path: Path,
    lr0: float,
    weight_decay: float,
    imgsz: int,
    optimizer: str,
    epochs: int,
    batch: int,
    cos_lr: bool,
    device: str,
) -> dict:
    """Train one configuration and return metrics dict."""
    from ultralytics import YOLO

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"sweep__{Path(model).stem}__"
        f"lr{lr0:.0e}__wd{weight_decay:.0e}__"
        f"img{imgsz}__{optimizer}__ep{epochs}__{timestamp}"
    )
    print(f"\n{'='*70}")
    print(f"  Starting: {run_name}")
    print(f"{'='*70}")

    train_kwargs = {
        **FIXED_AUGMENTATION,
        "data":         str(data_path),
        "epochs":       epochs,
        "batch":        batch,
        "imgsz":        imgsz,
        "lr0":          lr0,
        "weight_decay": weight_decay,
        "optimizer":    optimizer,
        "cos_lr":       cos_lr,
        "patience":     epochs,   # no early stop in sweep — run full epochs
        "device":       device,
        "project":      str(RUNS_DIR),
        "name":         run_name,
        "exist_ok":     True,
        "verbose":      False,    # suppress per-epoch output to keep logs clean
        "seed":         42,
    }

    yolo = YOLO(model)
    t0 = time.perf_counter()
    train_results = yolo.train(**train_kwargs)
    elapsed = time.perf_counter() - t0

    actual_run_dir = Path(train_results.save_dir).resolve()
    metrics = read_best_val_metrics(actual_run_dir)

    row = {
        "run_name":      run_name,
        "model":         Path(model).stem,
        "lr0":           lr0,
        "weight_decay":  weight_decay,
        "imgsz":         imgsz,
        "optimizer":     optimizer,
        "epochs":        epochs,
        "batch":         batch,
        "cos_lr":        cos_lr,
        "train_seconds": round(elapsed, 1),
        "timestamp":     timestamp,
        **metrics,
    }
    append_csv(row)
    print(f"  mAP@50={metrics.get('val_mAP50', 'N/A'):.4f}  "
          f"mAP@50-95={metrics.get('val_mAP50_95', 'N/A'):.4f}  "
          f"({elapsed/60:.1f} min)")
    return row


def print_summary():
    """Print ranked results table from the CSV after the sweep."""
    if not SWEEP_CSV.exists():
        return
    with SWEEP_CSV.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return

    def _f(r, k):
        try:
            return float(r.get(k, 0))
        except (ValueError, TypeError):
            return 0.0

    rows.sort(key=lambda r: _f(r, "val_mAP50"), reverse=True)
    header = f"{'Run':<55} {'LR':>7} {'WD':>8} {'Img':>5} {'Opt':>6} {'mAP50':>7} {'mAP50-95':>9}"
    print("\n" + "="*len(header))
    print("HYPERPARAMETER SWEEP RESULTS (ranked by val mAP@50)")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    for r in rows[:15]:
        print(
            f"{r['run_name'][:55]:<55} "
            f"{_f(r,'lr0'):>7.4f} "
            f"{_f(r,'weight_decay'):>8.5f} "
            f"{r.get('imgsz','?'):>5} "
            f"{r.get('optimizer','?'):>6} "
            f"{_f(r,'val_mAP50'):>7.4f} "
            f"{_f(r,'val_mAP50_95'):>9.4f}"
        )
    print(f"\nFull results → {SWEEP_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Poles2025.")
    parser.add_argument("--data",   type=Path,
                        default=REPO_ROOT / "data" / "combined" / "roadpoles_v1_plus_iphone" / "data.yaml")
    parser.add_argument("--model",  default="yolo26s",
                        help="Model name / path to start from (default: yolo26s)")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Epochs per sweep run (default: 35)")
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--cos-lr", action="store_true",
                        help="Enable cosine LR annealing for all sweep runs")
    parser.add_argument("--device", default=None,
                        help="Device override (e.g. '0', 'cpu'). Auto-detected if omitted.")
    args = parser.parse_args()

    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sweep grid: {json.dumps(SWEEP_GRID, indent=2)}")
    n_total = 1
    for v in SWEEP_GRID.values():
        n_total *= len(v)
    print(f"Total configurations: {n_total}\n")

    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    all_configs = list(itertools.product(*values))

    for i, config_vals in enumerate(all_configs, 1):
        cfg = dict(zip(keys, config_vals))
        print(f"\n[{i}/{n_total}] Config: {cfg}")
        run_single(
            model=args.model,
            data_path=args.data,
            lr0=cfg["lr0"],
            weight_decay=cfg["weight_decay"],
            imgsz=cfg["imgsz"],
            optimizer=cfg["optimizer"],
            epochs=args.epochs,
            batch=args.batch,
            cos_lr=args.cos_lr,
            device=device,
        )

    print_summary()


if __name__ == "__main__":
    main()
