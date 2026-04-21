"""
Build a leaderboard submission for the Road_poles_iPhone Snow Pole Detection page.

The leaderboard expects a zipped folder of YOLO .txt files (one per test image),
with the format:
    class cx cy w h confidence
where cx, cy, w, h are normalized to [0, 1] and `confidence` is the per-box score.

Usage:
    # Point at the test image folder directly:
    python scripts/make_submission.py \
        --weights runs/<final>/weights/best.pt \
        --test-dir /work/datasets/tdt4265/Poles2025/Road_poles_iPhone/images/Test/test

    # Or via a data yaml (uses its `test:` field):
    python scripts/make_submission.py \
        --weights runs/<final>/weights/best.pt \
        --data config/data_road_poles_iphone_cybele.yaml

Outputs:
    results/submissions/<tag>/labels/*.txt     (one per test image)
    results/submissions/<tag>.zip              (upload this to the leaderboard)
"""

import argparse
import shutil
import time
import zipfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = REPO_ROOT / "results" / "submissions"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_test_dir(data_yaml: Path) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", "")) if cfg.get("path") else data_yaml.parent
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    test_rel = cfg.get("test")
    if not test_rel:
        raise ValueError(f"No `test:` field in {data_yaml}")
    test_path = Path(test_rel)
    return test_path if test_path.is_absolute() else (root / test_path).resolve()


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def build_zip(labels_dir: Path, zip_path: Path) -> int:
    """Zip every .txt in labels_dir, flat (no parent folder). Return file count."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt in sorted(labels_dir.glob("*.txt")):
            zf.write(txt, arcname=txt.name)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Build Road_poles_iPhone leaderboard submission.")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to trained YOLO weights (best.pt).")
    parser.add_argument("--test-dir", type=Path, default=None,
                        help="Folder of test images. Overrides --data if given.")
    parser.add_argument("--data", type=Path, default=None,
                        help="Data yaml with a `test:` field (alternative to --test-dir).")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Low conf keeps all detections for mAP ranking (default: 0.001).")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU threshold (default: 0.6).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None,
                        help="Short label for the submission folder/zip (default: weights stem + timestamp).")
    parser.add_argument("--no-empty-files", action="store_true",
                        help="Do not create empty .txt files for images with no detections.")
    args = parser.parse_args()

    if args.test_dir is None and args.data is None:
        parser.error("Provide either --test-dir or --data.")

    test_dir = (args.test_dir.resolve() if args.test_dir
                else resolve_test_dir(args.data.resolve()))
    if not test_dir.exists():
        parser.error(f"Test dir does not exist: {test_dir}")

    images = list_images(test_dir)
    if not images:
        parser.error(f"No images found in {test_dir}")

    tag = args.tag or f"{args.weights.stem}__{time.strftime('%Y%m%d-%H%M%S')}"
    sub_root = SUB_DIR / tag
    if sub_root.exists():
        shutil.rmtree(sub_root)
    sub_root.mkdir(parents=True)

    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Weights    : {args.weights}")
    print(f"Test dir   : {test_dir}  ({len(images)} images)")
    print(f"conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    t0 = time.perf_counter()
    # Ultralytics writes predictions to <project>/<name>/labels/*.txt
    model.predict(
        source=str(test_dir),
        project=str(sub_root),
        name="predict",
        save_txt=True,
        save_conf=True,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=False,       # no annotated images needed for submission
        verbose=False,
        exist_ok=True,
    )
    infer_s = time.perf_counter() - t0

    labels_src = sub_root / "predict" / "labels"
    labels_out = sub_root / "labels"
    labels_out.mkdir(exist_ok=True)

    # Copy predicted .txt files into the flat labels_out dir
    copied = 0
    if labels_src.exists():
        for txt in labels_src.glob("*.txt"):
            shutil.copy2(txt, labels_out / txt.name)
            copied += 1

    # Create empty .txt files for images with no detections
    empties = 0
    if not args.no_empty_files:
        for img in images:
            txt = labels_out / f"{img.stem}.txt"
            if not txt.exists():
                txt.touch()
                empties += 1

    # Build the zip
    zip_path = SUB_DIR / f"{tag}.zip"
    n_in_zip = build_zip(labels_out, zip_path)
    zip_mib = zip_path.stat().st_size / (1024 * 1024)

    print(f"\nInference  : {infer_s:.1f}s  ({infer_s / len(images) * 1000:.1f} ms/img)")
    print(f"Predictions: {copied} files with detections  +  {empties} empty files")
    print(f"Zip entries: {n_in_zip}")
    print(f"Zip size   : {zip_mib:.3f} MiB   (leaderboard limit: 1.0 MiB)")
    if zip_mib > 1.0:
        print("WARNING: Zip exceeds 1.0 MiB. Raise --conf (e.g. 0.01 or 0.05) to shrink.")
    print(f"\nSubmission zip: {zip_path}")
    print(f"Upload this file to the Road poles iPhone leaderboard.")


if __name__ == "__main__":
    main()
