"""
Rebuild results/model_comparison.csv from the actual runs/compare__*/ folders.

The previous CSV was populated with best_epoch=1 for every model because the
summariser sorted by a `fitness` column that Ultralytics ≥ 8.4 no longer writes,
so every row scored -1.0 and the first epoch "won". This script re-reads each
run's results.csv + args.yaml and picks the best epoch by Ultralytics' standard
fitness formula:

    fitness = 0.1 * mAP@50 + 0.9 * mAP@0.5:0.95

Only completed runs (containing a results.csv) are included; the earlier
aborted launches in runs/compare__*/20260421-1406* are skipped.
"""

import csv
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
OUT_CSV = REPO_ROOT / "results" / "model_comparison.csv"

DESCRIPTIONS = {
    "yolo26n":  "YOLO26 Nano   — smallest, fastest; baseline reference",
    "yolo26s":  "YOLO26 Small  — mid-size YOLO26 variant",
    "yolo26m":  "YOLO26 Medium — accuracy vs. speed trade-off check",
    "yolo11s":  "YOLO11 Small  — prior-gen comparison (cross-architecture)",
}

FIELDS = [
    "rank", "model", "description",
    "val_precision", "val_recall", "val_mAP50", "val_mAP50_95",
    "best_epoch", "total_epochs", "train_hours",
    "lr0", "weight_decay", "imgsz", "optimizer", "data",
    "weights_path", "run_dir", "timestamp",
]


def fitness(mAP50: float, mAP50_95: float) -> float:
    return 0.1 * mAP50 + 0.9 * mAP50_95


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_run(run_dir: Path) -> dict | None:
    results_csv = run_dir / "results.csv"
    args_yaml   = run_dir / "args.yaml"
    weights     = run_dir / "weights" / "best.pt"
    if not results_csv.exists():
        return None

    with results_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    # Pick best epoch by fitness
    def row_fit(r):
        m50    = _f(r.get("metrics/mAP50(B)"))
        m50_95 = _f(r.get("metrics/mAP50-95(B)"))
        if m50 is None or m50_95 is None:
            return -1.0
        return fitness(m50, m50_95)

    best = max(rows, key=row_fit)

    # train_hours = last epoch's cumulative "time" (seconds) / 3600
    last_time = _f(rows[-1].get("time"))
    train_hours = round(last_time / 3600.0, 3) if last_time else None

    args = {}
    if args_yaml.exists():
        args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}

    model_name = Path(str(args.get("model", ""))).stem or run_dir.name.split("__")[1]

    # Timestamp is the trailing yyyymmdd-hhmmss chunk of the run name
    timestamp = run_dir.name.rsplit("__", 1)[-1]

    return {
        "model":          model_name,
        "description":    DESCRIPTIONS.get(model_name, ""),
        "val_precision":  round(_f(best.get("metrics/precision(B)")) or 0.0, 5),
        "val_recall":     round(_f(best.get("metrics/recall(B)")) or 0.0, 5),
        "val_mAP50":      round(_f(best.get("metrics/mAP50(B)")) or 0.0, 5),
        "val_mAP50_95":   round(_f(best.get("metrics/mAP50-95(B)")) or 0.0, 5),
        "best_epoch":     int(_f(best.get("epoch")) or 0),
        "total_epochs":   int(_f(rows[-1].get("epoch")) or 0),
        "train_hours":    train_hours,
        "lr0":            args.get("lr0"),
        "weight_decay":   args.get("weight_decay"),
        "imgsz":          args.get("imgsz"),
        "optimizer":      args.get("optimizer"),
        "data":           Path(str(args.get("data", ""))).name,
        "weights_path":   str(weights),
        "run_dir":        str(run_dir),
        "timestamp":      timestamp,
    }


def main():
    compare_dirs = sorted(RUNS_DIR.glob("compare__*"))
    rows: list[dict] = []
    for d in compare_dirs:
        row = parse_run(d)
        if row is None:
            print(f"[skip] {d.name}  (no results.csv)")
            continue
        print(f"[ ok ] {d.name}  mAP50={row['val_mAP50']:.4f}  mAP50-95={row['val_mAP50_95']:.4f}")
        rows.append(row)

    if not rows:
        print("No runs with results.csv found.")
        return

    # Prefer the run with the most completed epochs when the same model has duplicates
    best_per_model: dict[str, dict] = {}
    for r in rows:
        prev = best_per_model.get(r["model"])
        if prev is None or (r["total_epochs"] > prev["total_epochs"]):
            best_per_model[r["model"]] = r
    rows = list(best_per_model.values())

    # Rank by fitness (mAP50-95 weighted)
    rows.sort(
        key=lambda r: fitness(r["val_mAP50"], r["val_mAP50_95"]),
        reverse=True,
    )
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Print ranking table
    print("\n" + "=" * 95)
    print(f"{'rk':>2}  {'model':<9}  {'mAP50':>7}  {'mAP50-95':>9}  "
          f"{'P':>6}  {'R':>6}  {'best':>5}/{'tot':<4}  {'hrs':>5}  fitness")
    print("-" * 95)
    for r in rows:
        fit = fitness(r["val_mAP50"], r["val_mAP50_95"])
        print(
            f"{r['rank']:>2}  {r['model']:<9}  "
            f"{r['val_mAP50']:>7.4f}  {r['val_mAP50_95']:>9.4f}  "
            f"{r['val_precision']:>6.4f}  {r['val_recall']:>6.4f}  "
            f"{r['best_epoch']:>5}/{r['total_epochs']:<4}  "
            f"{(r['train_hours'] or 0):>5.2f}  {fit:.4f}"
        )
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
