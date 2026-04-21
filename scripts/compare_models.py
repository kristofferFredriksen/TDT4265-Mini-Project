"""
Model architecture comparison for Poles2025 snow pole detection.

Trains and evaluates four architectures under identical conditions so the
results are directly comparable for the project report:

  1. yolo26n   — nano, already-available baseline (~2 M params)
  2. yolo26s   — small, best single-run result so far
  3. yolo26m   — medium, checks whether accuracy gain justifies extra cost
  4. yolo11s   — previous-generation small, cross-generation comparison
  5. rtdetr-l  — transformer-based (optional; skip if memory-limited)

All models are trained with:
  - Augmentation from config/augmentation.yaml  (data-driven choices)
  - Best hyperparameters identified by scripts/hyper_sweep.py
    Override via --lr / --imgsz / --optimizer CLI flags.
  - 100 epochs with patience=20 and cos_lr=True
  - close_mosaic=15 (final 15 epochs without mosaic)

Results are written to  results/model_comparison.csv.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --epochs 80 --imgsz 960 --lr 0.001
    python scripts/compare_models.py --models yolo26s yolo26m --skip-rtdetr
"""

import argparse
import csv
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = REPO_ROOT / "runs"
COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"

DEFAULT_MODELS = [
    ("yolo26n",   "YOLO26 Nano   — smallest, fastest; baseline reference"),
    ("yolo26s",   "YOLO26 Small  — best architecture found in earlier runs"),
    ("yolo26m",   "YOLO26 Medium — accuracy vs. speed trade-off check"),
    ("yolo11s",   "YOLO11 Small  — prior-gen comparison (cross-architecture)"),
]
RTDETR_MODEL = ("rtdetr-l", "RT-DETR Large — transformer-based (optional)")

AUGMENTATION = {
    "hsv_h": 0.015, "hsv_s": 0.7,  "hsv_v": 0.5,
    "degrees": 5.0, "translate": 0.15, "scale": 0.6,
    "shear": 0.0,   "perspective": 0.0,
    "flipud": 0.0,  "fliplr": 0.5,
    "mosaic": 1.0,  "close_mosaic": 15,
    "mixup": 0.05,  "copy_paste": 0.1,
    "erasing": 0.4,
}

CSV_FIELDNAMES = [
    "rank", "model", "description",
    "val_precision", "val_recall", "val_mAP50", "val_mAP50_95",
    "best_epoch", "train_hours",
    "lr0", "weight_decay", "imgsz", "optimizer", "epochs",
    "weights_path", "timestamp",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_best_val_metrics(run_dir: Path) -> dict:
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
        "best_epoch":    int(_f(best, "epoch")),
    }


def write_csv(rows: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def train_model(
    model_name: str,
    description: str,
    data_path: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    lr0: float,
    weight_decay: float,
    optimizer: str,
    device: str,
) -> dict:
    """Train one model and return a results dict. Returns error info on failure."""
    from ultralytics import YOLO

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = model_name.replace("/", "-").replace(".", "_")
    run_name = f"compare__{safe_name}__img{imgsz}__ep{epochs}__{timestamp}"

    print(f"\n{'='*70}")
    print(f"  Model : {model_name}  ({description})")
    print(f"  Run   : {run_name}")
    print(f"{'='*70}")

    try:
        yolo = YOLO(model_name)
    except Exception as exc:
        print(f"  [SKIP] Could not load model '{model_name}': {exc}")
        return {
            "model": model_name, "description": description,
            "val_mAP50": -1, "val_mAP50_95": -1,
            "error": str(exc), "timestamp": timestamp,
        }

    train_kwargs = {
        **AUGMENTATION,
        "data":         str(data_path),
        "epochs":       epochs,
        "batch":        batch,
        "imgsz":        imgsz,
        "lr0":          lr0,
        "lrf":          0.01,
        "weight_decay": weight_decay,
        "optimizer":    optimizer,
        "cos_lr":       True,
        "warmup_epochs": 5,
        "patience":     20,
        "device":       device,
        "project":      str(RUNS_DIR),
        "name":         run_name,
        "exist_ok":     True,
        "verbose":      False,
        "seed":         42,
    }

    t0 = time.perf_counter()
    try:
        train_results = yolo.train(**train_kwargs)
    except Exception as exc:
        print(f"  [ERROR] Training failed: {exc}")
        return {
            "model": model_name, "description": description,
            "val_mAP50": -1, "val_mAP50_95": -1,
            "error": str(exc), "timestamp": timestamp,
        }

    elapsed_h = (time.perf_counter() - t0) / 3600.0
    actual_run_dir = Path(train_results.save_dir).resolve()
    metrics = read_best_val_metrics(actual_run_dir)
    best_weights = actual_run_dir / "weights" / "best.pt"

    row = {
        "model":        model_name,
        "description":  description,
        "lr0":          lr0,
        "weight_decay": weight_decay,
        "imgsz":        imgsz,
        "optimizer":    optimizer,
        "epochs":       epochs,
        "train_hours":  round(elapsed_h, 3),
        "weights_path": str(best_weights),
        "timestamp":    timestamp,
        **metrics,
    }
    print(
        f"  mAP@50={metrics.get('val_mAP50', 'N/A'):.4f}  "
        f"mAP@50-95={metrics.get('val_mAP50_95', 'N/A'):.4f}  "
        f"({elapsed_h*60:.1f} min)"
    )
    return row


def print_comparison_table(rows: list):
    """Print a ranked ASCII table to stdout."""
    valid = [r for r in rows if r.get("val_mAP50", -1) >= 0]
    valid.sort(key=lambda r: r.get("val_mAP50", 0), reverse=True)
    for rank, r in enumerate(valid, 1):
        r["rank"] = rank

    sep = "="*80
    print(f"\n{sep}")
    print("MODEL ARCHITECTURE COMPARISON  (ranked by val mAP@50)")
    print(sep)
    print(f"{'#':>2}  {'Model':<12} {'mAP@50':>7} {'mAP@50-95':>10} {'P':>7} {'R':>7} {'Time(h)':>8}  Description")
    print("-"*80)
    for r in valid:
        print(
            f"{r['rank']:>2}  "
            f"{r['model']:<12} "
            f"{r.get('val_mAP50', 0):>7.4f} "
            f"{r.get('val_mAP50_95', 0):>10.4f} "
            f"{r.get('val_precision', 0):>7.4f} "
            f"{r.get('val_recall', 0):>7.4f} "
            f"{r.get('train_hours', 0):>8.3f}  "
            f"{r.get('description', '')}"
        )
    print(f"\nFull results → {COMPARISON_CSV}")

    if valid:
        best = valid[0]
        print(f"\nBest model: {best['model']}  mAP@50={best.get('val_mAP50',0):.4f}")
        print(f"Best weights: {best.get('weights_path', 'N/A')}")
        print("Pass this weights path to scripts/train_final.py via --weights")


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO architectures on Poles2025.")
    parser.add_argument("--data",   type=Path,
                        default=REPO_ROOT / "config" / "data_road_poles_iphone_cybele.yaml")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model names to run (default: all DEFAULT_MODELS)")
    parser.add_argument("--epochs", type=int,  default=100)
    parser.add_argument("--batch",  type=int,  default=8)
    parser.add_argument("--imgsz",  type=int,  default=960,
                        help="Image size (use best from hyper_sweep, default 960)")
    parser.add_argument("--lr",     type=float, default=0.001,
                        help="Learning rate (use best from hyper_sweep)")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--include-rtdetr", action="store_true",
                        help="Also train RT-DETR-l (slow; ~4× longer than yolo26m)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_list = list(DEFAULT_MODELS)
    if args.include_rtdetr:
        model_list.append(RTDETR_MODEL)

    if args.models:
        allowed = set(args.models)
        model_list = [(m, d) for m, d in model_list if m in allowed]

    print(f"Models to compare: {[m for m, _ in model_list]}")
    print(f"Hyperparameters : lr={args.lr}  wd={args.weight_decay}  img={args.imgsz}  opt={args.optimizer}")

    all_rows = []
    for model_name, description in model_list:
        row = train_model(
            model_name=model_name,
            description=description,
            data_path=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            device=device,
        )
        all_rows.append(row)
        # Write after every model so partial results survive crashes
        write_csv(all_rows)

    print_comparison_table(all_rows)


if __name__ == "__main__":
    main()
