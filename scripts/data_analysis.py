"""
Data analysis for Poles2025 snow pole detection dataset.

Reads YOLO-format labels from all available splits and produces:
  - Image/annotation counts per split
  - Bounding-box width, height, and aspect-ratio distributions
  - Heatmap of box center positions (x, y)
  - Distribution of pole count per image

All plots are saved to  <repo_root>/analysis/.

Usage:
    python scripts/data_analysis.py
    python scripts/data_analysis.py --data config/data_roadpoles_v1_cybele.yaml
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless backend for cluster use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "analysis"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_split_dirs(data_cfg: dict, data_cfg_path: Path):
    """Return {split_name: (images_dir, labels_dir)} for all splits in the config."""
    dataset_root_raw = data_cfg.get("path", "")
    dataset_root = Path(dataset_root_raw)
    if not dataset_root.is_absolute():
        dataset_root = (data_cfg_path.parent / dataset_root).resolve()

    splits = {}
    for split in ("train", "val", "valid", "test"):
        if split not in data_cfg:
            continue
        split_images_raw = data_cfg[split]
        images_dir = Path(split_images_raw)
        if not images_dir.is_absolute():
            images_dir = (dataset_root / images_dir).resolve()
        labels_dir = images_dir.parent.parent / "labels" if "images" in images_dir.parts else images_dir.parent / "labels"
        # normalise: replace trailing 'images' folder with 'labels'
        if images_dir.name == "images":
            labels_dir = images_dir.parent / "labels"
        else:
            labels_dir = images_dir.parent / "labels"
        key = "val" if split == "valid" else split
        splits[key] = (images_dir, labels_dir)
    return splits


def load_labels(labels_dir: Path):
    """Parse all *.txt YOLO label files; return list of (cx, cy, w, h) arrays."""
    if not labels_dir.exists():
        return [], 0

    all_boxes = []
    n_images = 0
    boxes_per_image = []
    for txt in sorted(labels_dir.glob("*.txt")):
        n_images += 1
        boxes = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = parts
                boxes.append([float(cx), float(cy), float(w), float(h)])
        all_boxes.extend(boxes)
        boxes_per_image.append(len(boxes))
    return np.array(all_boxes) if all_boxes else np.empty((0, 4)), n_images, boxes_per_image


def count_images(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in exts)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def savefig(fig, name: str):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bbox_distributions(splits_data: dict):
    """Histogram of bbox width, height, and aspect ratio per split."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Bounding-box size distributions (normalised to image size)", fontsize=13)

    colors = plt.cm.tab10.colors
    for idx, (split, (boxes, _, _)) in enumerate(splits_data.items()):
        if boxes.shape[0] == 0:
            continue
        c = colors[idx % len(colors)]
        w, h = boxes[:, 2], boxes[:, 3]
        ar = w / np.maximum(h, 1e-6)
        axes[0].hist(w, bins=60, alpha=0.6, color=c, label=split)
        axes[1].hist(h, bins=60, alpha=0.6, color=c, label=split)
        axes[2].hist(ar, bins=60, alpha=0.6, color=c, label=split)

    axes[0].set_xlabel("Box width (normalised)")
    axes[1].set_xlabel("Box height (normalised)")
    axes[2].set_xlabel("Aspect ratio  w/h")
    for ax in axes:
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    savefig(fig, "bbox_distributions.png")


def plot_center_heatmaps(splits_data: dict):
    """2-D heatmap of box centre positions for each split."""
    n = len(splits_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Box centre position heatmap (normalised image coordinates)", fontsize=13)

    for ax, (split, (boxes, _, _)) in zip(axes, splits_data.items()):
        if boxes.shape[0] == 0:
            ax.set_title(f"{split} (no data)")
            continue
        cx, cy = boxes[:, 0], boxes[:, 1]
        h2d, xedges, yedges = np.histogram2d(cx, cy, bins=40, range=[[0, 1], [0, 1]])
        ax.imshow(
            h2d.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
            cmap="hot",
        )
        ax.set_title(f"{split}  (n={boxes.shape[0]})")
        ax.set_xlabel("x (left→right)")
        ax.set_ylabel("y (top→bottom)")
    savefig(fig, "center_heatmaps.png")


def plot_poles_per_image(splits_data: dict):
    """Bar chart: distribution of number of poles per image."""
    fig, axes = plt.subplots(1, len(splits_data), figsize=(5 * len(splits_data), 4), squeeze=False)
    fig.suptitle("Number of poles per image", fontsize=13)

    for ax, (split, (_, _, bpi)) in zip(axes[0], splits_data.items()):
        if not bpi:
            ax.set_title(f"{split} (no data)")
            continue
        max_count = max(bpi)
        bins = np.arange(0, max_count + 2) - 0.5
        ax.hist(bpi, bins=bins, edgecolor="black", color="steelblue", alpha=0.8)
        ax.set_title(f"{split}  (images={len(bpi)})")
        ax.set_xlabel("Poles per image")
        ax.set_ylabel("Image count")
        ax.set_xticks(range(0, max_count + 1))
    savefig(fig, "poles_per_image.png")


def plot_combined_summary(splits_data: dict, counts: dict):
    """One-page summary figure with all key stats."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Poles2025 — Dataset Analysis Summary", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
    split_list = list(splits_data.keys())

    # ---- Row 0: width / height / aspect-ratio histograms (span 3 cols each row)
    ax_w   = fig.add_subplot(gs[0, 0])
    ax_h   = fig.add_subplot(gs[0, 1])
    ax_ar  = fig.add_subplot(gs[0, 2])
    ax_cnt = fig.add_subplot(gs[0, 3])

    for split, (boxes, n_img, bpi) in splits_data.items():
        if boxes.shape[0] == 0:
            continue
        c = colors.get(split, "grey")
        w, h = boxes[:, 2], boxes[:, 3]
        ar = w / np.maximum(h, 1e-6)
        ax_w.hist(w,  bins=50, alpha=0.65, color=c, label=split)
        ax_h.hist(h,  bins=50, alpha=0.65, color=c, label=split)
        ax_ar.hist(ar, bins=50, alpha=0.65, color=c, label=split)

    ax_w.set_title("Box width");  ax_w.set_xlabel("width (norm)");   ax_w.legend(fontsize=7)
    ax_h.set_title("Box height"); ax_h.set_xlabel("height (norm)");  ax_h.legend(fontsize=7)
    ax_ar.set_title("Aspect ratio (w/h)"); ax_ar.set_xlabel("w/h"); ax_ar.legend(fontsize=7)

    # Image counts bar
    split_names = list(counts.keys())
    img_vals  = [counts[s]["images"]  for s in split_names]
    ann_vals  = [counts[s]["annotations"] for s in split_names]
    x = np.arange(len(split_names))
    ax_cnt.bar(x - 0.2, img_vals, 0.4, label="images", color="steelblue")
    ax_cnt.bar(x + 0.2, ann_vals, 0.4, label="annotations", color="darkorange")
    ax_cnt.set_xticks(x); ax_cnt.set_xticklabels(split_names)
    ax_cnt.set_title("Dataset counts"); ax_cnt.legend(fontsize=7)

    # ---- Row 1: heatmaps for each split (up to 4)
    for col, split in enumerate(split_list[:4]):
        boxes, n_img, _ = splits_data[split]
        ax = fig.add_subplot(gs[1, col])
        if boxes.shape[0] == 0:
            ax.set_title(f"{split}\n(no labels)")
            continue
        cx, cy = boxes[:, 0], boxes[:, 1]
        h2d, _, _ = np.histogram2d(cx, cy, bins=30, range=[[0, 1], [0, 1]])
        ax.imshow(h2d.T, origin="lower", extent=[0, 1, 0, 1],
                  aspect="equal", cmap="YlOrRd")
        ax.set_title(f"{split} centre heatmap\nn={boxes.shape[0]}")
        ax.set_xlabel("x →"); ax.set_ylabel("y ↓")

    savefig(fig, "summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse Poles2025 dataset statistics.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "combined" / "roadpoles_v1_plus_iphone" / "data.yaml",
        help="Path to dataset YAML (YOLO format).",
    )
    args = parser.parse_args()

    data_cfg_path = args.data.resolve()
    if not data_cfg_path.exists():
        # fall back to cybele config
        data_cfg_path = REPO_ROOT / "config" / "data_roadpoles_v1_cybele.yaml"
    data_cfg = load_yaml(data_cfg_path)
    print(f"Dataset config : {data_cfg_path}")

    split_dirs = resolve_split_dirs(data_cfg, data_cfg_path)

    splits_data = {}
    counts = {}
    for split, (img_dir, lbl_dir) in split_dirs.items():
        n_img_files = count_images(img_dir)
        result = load_labels(lbl_dir)
        if len(result) == 3:
            boxes, n_lbl_imgs, bpi = result
        else:
            boxes, n_lbl_imgs, bpi = np.empty((0, 4)), 0, []

        n_annotations = boxes.shape[0]
        splits_data[split] = (boxes, n_lbl_imgs, bpi)
        counts[split] = {"images": n_img_files or n_lbl_imgs, "annotations": n_annotations}

        print(f"\n  [{split}]")
        print(f"    Images          : {counts[split]['images']}")
        print(f"    Label files     : {n_lbl_imgs}")
        print(f"    Total poles     : {n_annotations}")
        if n_lbl_imgs:
            print(f"    Avg poles/image : {n_annotations / n_lbl_imgs:.2f}")
        if boxes.shape[0]:
            w, h = boxes[:, 2], boxes[:, 3]
            print(f"    Box width  — mean={w.mean():.4f}  median={np.median(w):.4f}  p95={np.percentile(w,95):.4f}")
            print(f"    Box height — mean={h.mean():.4f}  median={np.median(h):.4f}  p95={np.percentile(h,95):.4f}")
            ar = w / np.maximum(h, 1e-6)
            print(f"    Aspect ratio (w/h) — mean={ar.mean():.3f}  median={np.median(ar):.3f}")

    print("\nGenerating plots …")
    plot_bbox_distributions(splits_data)
    plot_center_heatmaps(splits_data)
    plot_poles_per_image(splits_data)
    plot_combined_summary(splits_data, counts)

    print(f"\nAll plots saved to: {ANALYSIS_DIR}")
    print("\n--- Insights for augmentation design ---")
    all_boxes = np.concatenate([v[0] for v in splits_data.values() if v[0].shape[0] > 0], axis=0)
    if all_boxes.shape[0]:
        w, h = all_boxes[:, 2], all_boxes[:, 3]
        print(f"  Median pole width  : {np.median(w)*100:.1f}% of image width  → high-res training critical")
        print(f"  Median pole height : {np.median(h)*100:.1f}% of image height")
        print(f"  Poles w < 2%       : {(w < 0.02).mean()*100:.0f}% of instances  → copy_paste augmentation helps")
        print(f"  Aspect ratio (w/h) < 0.3 : {(w/np.maximum(h,1e-6) < 0.3).mean()*100:.0f}%  → vertical thin objects")
        cx = all_boxes[:, 0]
        print(f"  Horizontal spread  : std={cx.std():.3f}  → translate augmentation worthwhile")


if __name__ == "__main__":
    main()
