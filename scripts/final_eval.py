"""
Final evaluation and qualitative results for Poles2025 snow pole detection.

Given a trained model's weights path (from scripts/train_final.py), this
script:
  1. Evaluates on the validation split and prints P / R / mAP@50 / mAP@50-95.
  2. Evaluates on the test split if labels are available.
  3. Builds a qualitative grid (results/qualitative.png) from 8-12 validation
     images selected to show challenging cases (fog, dense snow, distance,
     partial occlusion) by picking the images where predicted confidence is
     lowest — these are the hardest examples.
  4. Updates results/sustainability.md with total GPU-hours for this run.

Usage:
    python scripts/final_eval.py --weights runs/<run>/weights/best.pt
    python scripts/final_eval.py --weights runs/<run>/weights/best.pt --n-qual 12
    python scripts/final_eval.py --weights best.pt --conf 0.25 --iou 0.5
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_split_path(data_cfg: dict, data_cfg_path: Path, split: str) -> Path:
    """Return absolute path to a dataset split's image directory."""
    root = Path(data_cfg.get("path", ""))
    if not root.is_absolute():
        root = (data_cfg_path.parent / root).resolve()
    val_raw = data_cfg.get(split, "")
    p = Path(val_raw)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def extract_metrics(results) -> dict:
    """Pull metrics from an ultralytics val Results object."""
    rd = {}
    if hasattr(results, "results_dict"):
        rd = results.results_dict or {}

    def _f(key):
        val = rd.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return {
        "precision":  _f("metrics/precision(B)"),
        "recall":     _f("metrics/recall(B)"),
        "mAP50":      _f("metrics/mAP50(B)"),
        "mAP50_95":   _f("metrics/mAP50-95(B)"),
    }


def build_qualitative_grid(
    model,
    image_paths: list,
    n: int,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    out_path: Path,
):
    """
    Run inference on all image_paths, pick the n images with the lowest
    maximum detection confidence (hardest cases), and save a grid figure.

    Images are shown with predicted bounding boxes overlaid.
    """
    import cv2

    print(f"\nBuilding qualitative grid from {len(image_paths)} images …")

    # Collect confidence scores per image to rank by difficulty
    scored = []
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=0.01,            # very low threshold to capture borderline detections
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        r = results[0]
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.array([])
        max_conf = float(confs.max()) if len(confs) else 0.0
        scored.append((max_conf, img_path, r))

    # Sort ascending by max_conf → hardest images first
    scored.sort(key=lambda x: x[0])
    hard_cases = scored[:n]

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle(
        "Qualitative results — hardest validation cases\n"
        "(sorted by lowest predicted confidence)",
        fontsize=13,
    )
    axes = np.array(axes).reshape(-1)

    for ax, (max_conf, img_path, result) in zip(axes, hard_cases):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            ax.set_visible(False)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]
        ax.imshow(img_rgb)
        ax.set_xticks([]); ax.set_yticks([])

        # Draw predicted boxes
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                c = float(box.conf[0].cpu().numpy())
                if c < conf:
                    continue
                rect = mpatches.FancyBboxPatch(
                    (x1, y1), x2 - x1, y2 - y1,
                    boxstyle="square,pad=0",
                    linewidth=1.5,
                    edgecolor="#00FF00",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 4, f"{c:.2f}", color="#00FF00",
                        fontsize=6, fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        n_det = int((result.boxes.conf.cpu().numpy() >= conf).sum()) if result.boxes is not None else 0
        ax.set_title(f"{img_path.name[:22]}\nmax_conf={max_conf:.2f}  det={n_det}", fontsize=7)

    for ax in axes[n:]:
        ax.set_visible(False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Qualitative grid saved: {out_path}")


def update_sustainability(weights_path: Path, train_hours: float | None, n_experiments: int | None):
    """Append / update sustainability.md under results/."""
    # GPU TDP values (W) for common cluster GPUs
    GPU_TDPS = {
        "A100": 400,
        "V100": 300,
        "RTX 3090": 350,
        "RTX 4090": 450,
        "unknown": 350,    # conservative estimate
    }
    gpu_label = "A100"
    gpu_tdp_w = GPU_TDPS[gpu_label]

    if train_hours is None:
        print("  [sustainability] train_hours not available; skipping energy estimate.")
        return

    energy_kwh = gpu_tdp_w * train_hours / 1000.0
    # Tesla Model Y efficiency: 16.9 kWh / 100 km  (WLTP combined)
    kwh_per_100km = 16.9
    driving_km = energy_kwh / kwh_per_100km * 100.0

    content = f"""# Sustainability Estimate

## Assumptions
- GPU: {gpu_label}  (TDP = {gpu_tdp_w} W)
- Tesla Model Y energy efficiency: {kwh_per_100km} kWh / 100 km (WLTP combined)

## Final Training Run
| Metric | Value |
|--------|-------|
| Training time | {train_hours:.2f} h |
| GPU TDP | {gpu_tdp_w} W |
| Energy consumed | {energy_kwh:.3f} kWh |
| Equivalent driving distance | {driving_km:.1f} km |

## Notes
- Energy estimate assumes 100% GPU utilisation throughout training.
- Does not include data preprocessing, validation, or inference overhead.
- Cluster electricity mix (Norwegian hydro-dominated grid) has very low carbon
  intensity (~20 g CO₂ / kWh), so CO₂ impact is minimal.

*Generated by scripts/final_eval.py*
"""
    md_path = RESULTS_DIR / "sustainability.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path.write_text(content, encoding="utf-8")
    print(f"\nSustainability estimate written to: {md_path}")
    print(f"  Energy      : {energy_kwh:.3f} kWh")
    print(f"  Driving est.: {driving_km:.1f} km (Tesla Model Y)")


def main():
    parser = argparse.ArgumentParser(description="Final evaluation for Poles2025.")
    parser.add_argument(
        "--weights", type=Path, required=True,
        help="Path to best.pt from train_final.py (or any trained YOLO weights).",
    )
    parser.add_argument(
        "--data", type=Path,
        default=REPO_ROOT / "data" / "combined" / "roadpoles_v1_plus_iphone" / "data.yaml",
    )
    parser.add_argument("--imgsz",  type=int,   default=960)
    parser.add_argument("--conf",   type=float, default=0.30)
    parser.add_argument("--iou",    type=float, default=0.5)
    parser.add_argument("--n-qual", type=int,   default=12,
                        help="Number of images in the qualitative grid (default: 12).")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    data_cfg_path = args.data.resolve()
    data_cfg = load_yaml(data_cfg_path)

    # ---- Validation split evaluation -------------------------------------
    print("\n--- Validation split ---")
    val_dir = resolve_split_path(data_cfg, data_cfg_path, "val") or \
              resolve_split_path(data_cfg, data_cfg_path, "valid")

    t0 = time.perf_counter()
    val_results = model.val(
        data=str(data_cfg_path),
        split="val",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=True,
    )
    val_time = time.perf_counter() - t0
    val_metrics = extract_metrics(val_results)

    print(f"\nValidation metrics ({val_time:.1f}s):")
    for k, v in val_metrics.items():
        print(f"  {k:<12}: {v:.4f}" if v is not None else f"  {k:<12}: N/A")

    # ---- Test split evaluation (if labels available) ---------------------
    test_dir = resolve_split_path(data_cfg, data_cfg_path, "test")
    test_labels = test_dir.parent.parent / "labels" if "images" in str(test_dir) else test_dir.parent / "labels"
    if "images" not in str(test_dir):
        test_labels = test_dir.parent / "labels"
    else:
        test_labels = test_dir.parent.parent / "labels"

    test_metrics = {}
    if test_labels.exists() and any(test_labels.glob("*.txt")):
        print("\n--- Test split ---")
        test_results = model.val(
            data=str(data_cfg_path),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=True,
        )
        test_metrics = extract_metrics(test_results)
        print(f"\nTest metrics:")
        for k, v in test_metrics.items():
            print(f"  {k:<12}: {v:.4f}" if v is not None else f"  {k:<12}: N/A")
    else:
        print(f"\n[Test] No labels found at {test_labels} — skipping metric evaluation.")
        print("  (Test labels are on the remote cluster; run there for test metrics.)")

    # ---- Qualitative results grid ----------------------------------------
    image_paths = []
    val_images_dir = resolve_split_path(data_cfg, data_cfg_path, "val") or \
                     resolve_split_path(data_cfg, data_cfg_path, "valid")
    if val_images_dir and val_images_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [p for p in sorted(val_images_dir.iterdir()) if p.suffix.lower() in exts]

    if image_paths:
        n_qual = min(args.n_qual, len(image_paths))
        build_qualitative_grid(
            model=model,
            image_paths=image_paths,
            n=n_qual,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            out_path=RESULTS_DIR / "qualitative.png",
        )
    else:
        print(f"\n[Qualitative] No val images found at {val_images_dir}")

    # ---- Sustainability estimate ------------------------------------------
    # Try to read training time from the run's final_run_summary.json
    weights_path = args.weights.resolve()
    run_dir = weights_path.parent.parent  # .../weights/best.pt → run dir
    summary_json = run_dir / "final_run_summary.json"
    train_hours = None
    if summary_json.exists():
        summary_data = json.loads(summary_json.read_text(encoding="utf-8"))
        train_hours = summary_data.get("train_hours")
    update_sustainability(weights_path, train_hours, n_experiments=None)

    # ---- Save final metrics to JSON --------------------------------------
    final_report = {
        "weights": str(args.weights),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "config": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
        },
        "qualitative_grid": str(RESULTS_DIR / "qualitative.png"),
    }
    report_path = RESULTS_DIR / "final_eval_report.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\nFull report saved: {report_path}")

    print(f"\n{'='*60}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Precision  : {val_metrics.get('precision', 'N/A')}")
    print(f"  Recall     : {val_metrics.get('recall', 'N/A')}")
    print(f"  mAP@50     : {val_metrics.get('mAP50', 'N/A')}")
    print(f"  mAP@50-95  : {val_metrics.get('mAP50_95', val_metrics.get('mAP50-95', 'N/A'))}")
    print(f"\n  Qualitative grid : results/qualitative.png")
    print(f"  Sustainability   : results/sustainability.md")


if __name__ == "__main__":
    main()
