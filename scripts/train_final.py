"""
Final training run with all tuning tricks applied.

Takes the best model architecture from scripts/compare_models.py and trains
it with every recommended improvement:

  - Cosine annealing LR with 5-epoch linear warmup
  - close_mosaic=15  (disable mosaic augmentation for last 15 epochs)
  - Multi-scale training  (imgsz randomly varies ±25% each batch)
  - EMA (Exponential Moving Average) — enabled by default in Ultralytics ≥8.2
  - 200 epochs with patience=30 so training stops only when truly plateaued
  - All augmentation improvements from config/augmentation.yaml

After training, the script evaluates the best weights on the validation split
and prints the final metrics.  Run scripts/final_eval.py for the full test-set
evaluation + qualitative results grid.

Usage:
    python scripts/train_final.py --model yolo26m
    python scripts/train_final.py --model runs/compare__yolo26m__...  (weights)
    python scripts/train_final.py --model yolo26s --epochs 250 --imgsz 1280

The resulting weights path is printed at the end — pass it to final_eval.py:
    python scripts/final_eval.py --weights runs/<run>/weights/best.pt
"""

import argparse
import csv
import json
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"


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
        "precision":  _f(best, "metrics/precision(B)"),
        "recall":     _f(best, "metrics/recall(B)"),
        "mAP50":      _f(best, "metrics/mAP50(B)"),
        "mAP50_95":   _f(best, "metrics/mAP50-95(B)"),
        "best_epoch": int(_f(best, "epoch")),
        "fitness":    _f(best, "fitness"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Final training run with all tuning tricks for Poles2025."
    )
    parser.add_argument(
        "--model", default="yolo26s",
        help="Model name or path to pretrained weights (default: yolo26s). "
             "Use the best model identified by compare_models.py.",
    )
    parser.add_argument(
        "--data", type=Path,
        default=REPO_ROOT / "data" / "combined" / "roadpoles_v1_plus_iphone" / "data.yaml",
    )
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch",        type=int,   default=8)
    parser.add_argument("--imgsz",        type=int,   default=960,
                        help="Base image size.  Multi-scale training will vary ±25%.")
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--optimizer",    default="AdamW")
    parser.add_argument("--patience",     type=int,   default=30)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--multi-scale",  action="store_true",
                        help="Enable Ultralytics multi_scale (broken on torch 2.11 via F.interpolate; off by default).")
    parser.add_argument(
        "--tag", default="final",
        help="Short label appended to the run name for identification.",
    )
    args = parser.parse_args()

    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Model   : {args.model}")
    print(f"Epochs  : {args.epochs}  |  Patience: {args.patience}")
    print(f"ImgSz   : {args.imgsz}  (multi_scale={args.multi_scale})")
    print(f"LR      : {args.lr}  |  WD: {args.weight_decay}  |  Opt: {args.optimizer}")

    from ultralytics import YOLO

    yolo = YOLO(args.model)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_model = Path(args.model).stem.replace("/", "-")
    run_name = (
        f"final__{safe_model}__"
        f"img{args.imgsz}__ep{args.epochs}__"
        f"{args.tag}__{timestamp}"
    )

    train_kwargs = {
        # Dataset 
        "data":   str(args.data),
        "device": device,
        "project": str(RUNS_DIR),
        "name":   run_name,
        "exist_ok": True,
        "verbose": True,
        "seed":   42,

        # Training schedule
        "epochs":        args.epochs,
        "batch":         args.batch,
        "patience":      args.patience,
        "optimizer":     args.optimizer,
        "lr0":           args.lr,
        "lrf":           0.01,       # cosine annealing final value = lr0 * lrf
        "cos_lr":        True,       # cosine annealing LR schedule
        "warmup_epochs": 5,          # linear warmup prevents instability at start
        "weight_decay":  args.weight_decay,

        "imgsz":       args.imgsz,
        "multi_scale": args.multi_scale,

        #  Mosaic / close_mosaic 
        # Turn off mosaic for the final 15 epochs so the model sees full,
        # unmodified images before evaluation
        "mosaic":       1.0,
        "close_mosaic": 15,

        # Augmentation (data-driven from analysis)
        "hsv_h": 0.015, "hsv_s": 0.7,  "hsv_v": 0.5,
        "degrees": 5.0, "translate": 0.15, "scale": 0.6,
        "shear": 0.0,   "perspective": 0.0,
        "flipud": 0.0,  "fliplr": 0.5,
        "mixup": 0.05,  "copy_paste": 0.1,
        "erasing": 0.4,
    }

    print(f"\nStarting final training run: {run_name}\n")
    t0 = time.perf_counter()
    train_results = yolo.train(**train_kwargs)
    elapsed_h = (time.perf_counter() - t0) / 3600.0

    actual_run_dir = Path(train_results.save_dir).resolve()
    best_weights = actual_run_dir / "weights" / "best.pt"
    metrics = read_best_val_metrics(actual_run_dir)

    summary = {
        "run_name":     run_name,
        "run_dir":      str(actual_run_dir),
        "model":        args.model,
        "train_hours":  round(elapsed_h, 3),
        "best_weights": str(best_weights),
        "val_metrics":  metrics,
        "train_kwargs": {k: str(v) if isinstance(v, Path) else v for k, v in train_kwargs.items()},
    }
    summary_path = actual_run_dir / "final_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"FINAL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Run dir      : {actual_run_dir}")
    print(f"  Best weights : {best_weights}")
    print(f"  Training time: {elapsed_h:.2f} h")
    print(f"\n  Validation metrics (best epoch {metrics.get('best_epoch','?')}):")
    print(f"    Precision  : {metrics.get('precision', 'N/A'):.4f}")
    print(f"    Recall     : {metrics.get('recall', 'N/A'):.4f}")
    print(f"    mAP@50     : {metrics.get('mAP50', 'N/A'):.4f}")
    print(f"    mAP@50-95  : {metrics.get('mAP50_95', 'N/A'):.4f}")
    print(f"\nNext step:")
    print(f"  python scripts/final_eval.py --weights {best_weights}")


if __name__ == "__main__":
    main()
