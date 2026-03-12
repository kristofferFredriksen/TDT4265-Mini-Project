import argparse
import csv
import json
import logging
import re
import socket
import time
import zipfile
from pathlib import Path

import torch as t
import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = REPO_ROOT / "config" / "yolo_params.yml"
DEFAULT_MODEL = "yolo26n.pt"
RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_REGISTRY_PATH = RUNS_DIR / "experiment_registry.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a YOLO model for snow pole detection."
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_PARAMS_PATH,
        help="Path to the training parameter YAML file.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional override for the dataset YAML file.",
    )
    return parser.parse_args()


def resolve_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_existing_path(path_value, *base_dirs):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    for base_dir in base_dirs:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    return (base_dirs[0] / path).resolve()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(path, data):
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def setup_logger(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "python.log"
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_device():
    if t.cuda.is_available() and t.cuda.device_count() > 0:
        return "0"
    return "cpu"


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def extract_metrics(metrics):
    metrics_dict = to_builtin(getattr(metrics, "results_dict", {}) or {})
    return {
        "precision": metrics_dict.get("metrics/precision(B)"),
        "recall": metrics_dict.get("metrics/recall(B)"),
        "mAP50": metrics_dict.get("metrics/mAP50(B)"),
        "mAP50-95": metrics_dict.get("metrics/mAP50-95(B)"),
        "fitness": metrics_dict.get("fitness"),
        "raw": metrics_dict,
    }


def read_dataset_name(data_config):
    dataset_root = data_config.get("path")
    if dataset_root:
        return Path(dataset_root).name
    return "unknown_dataset"


def sanitize_component(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip("-") or "na"


def format_float_component(value):
    return format(float(value), ".0e").replace("+0", "").replace("+", "")


def build_run_name(dataset_name, params, timestamp):
    model_name = Path(params.get("model", DEFAULT_MODEL)).stem
    train_params = params["train"]
    components = [
        sanitize_component(dataset_name),
        sanitize_component(model_name),
        f"img{train_params['imgsz']}",
        f"ep{train_params['epochs']}",
        f"bs{train_params['batch']}",
    ]
    if train_params.get("lr0") is not None:
        components.append(f"lr{format_float_component(train_params['lr0'])}")
    run_tag = params.get("run_tag")
    if run_tag:
        components.append(sanitize_component(run_tag))
    components.append(timestamp)
    return "__".join(components)


def resolve_dataset_split(data_config_path, data_config, split_name):
    dataset_root = resolve_existing_path(
        data_config["path"],
        REPO_ROOT,
        data_config_path.parent,
    )
    split_path = Path(data_config[split_name])
    if split_path.is_absolute():
        return split_path
    return (dataset_root / split_path).resolve()


def split_has_labels(split_images_dir):
    split_dir = split_images_dir.parent
    labels_dir = split_dir / "labels"
    return labels_dir.exists() and any(labels_dir.glob("*.txt"))


def read_best_validation_metrics(run_dir):
    results_csv_path = run_dir / "results.csv"
    if not results_csv_path.exists():
        return None

    with results_csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        return None

    def parse_metric(row, key, default=-1.0):
        value = row.get(key, "")
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    best_row = max(rows, key=lambda row: parse_metric(row, "fitness"))
    return {
        "epoch": parse_metric(best_row, "epoch", default=None),
        "precision": parse_metric(best_row, "metrics/precision(B)", default=None),
        "recall": parse_metric(best_row, "metrics/recall(B)", default=None),
        "mAP50": parse_metric(best_row, "metrics/mAP50(B)", default=None),
        "mAP50-95": parse_metric(best_row, "metrics/mAP50-95(B)", default=None),
        "fitness": parse_metric(best_row, "fitness", default=None),
    }


def zip_submission(predictions_dir):
    labels_dir = predictions_dir / "labels"
    if not labels_dir.exists():
        return None

    zip_path = predictions_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label_file in sorted(labels_dir.glob("*.txt")):
            archive.write(label_file, arcname=label_file.name)
    return zip_path


def append_experiment_registry(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def normalize_params(params):
    model = params.get("model", DEFAULT_MODEL)
    train_defaults = {
        "epochs": params.get("epochs", 100),
        "batch": params.get("batch_size", 16),
        "imgsz": params.get("img_size", 640),
        "lr0": params.get("lr", 0.001),
        "patience": params.get("patience", 30),
        "optimizer": params.get("optimizer", "auto"),
        "weight_decay": params.get("weight_decay", 0.0005),
        "seed": params.get("seed", 0),
        "cos_lr": params.get("cos_lr", False),
        "degrees": params.get("degrees", 0.0),
        "translate": params.get("translate", 0.1),
        "scale": params.get("scale", 0.5),
        "shear": params.get("shear", 0.0),
        "perspective": params.get("perspective", 0.0),
        "flipud": params.get("flipud", 0.0),
        "fliplr": params.get("fliplr", 0.5),
        "mosaic": params.get("mosaic", 1.0),
        "mixup": params.get("mixup", 0.0),
        "copy_paste": params.get("copy_paste", 0.0),
        "erasing": params.get("erasing", 0.4),
        "hsv_h": params.get("hsv_h", 0.015),
        "hsv_s": params.get("hsv_s", 0.7),
        "hsv_v": params.get("hsv_v", 0.4),
        "close_mosaic": params.get("close_mosaic", 10),
        "verbose": params.get("verbose", True),
    }
    train_params = {**train_defaults, **params.get("train", {})}

    val_defaults = {
        "batch": train_params["batch"],
        "imgsz": train_params["imgsz"],
        "verbose": train_params.get("verbose", True),
    }
    val_params = {**val_defaults, **params.get("val", {})}

    predict_defaults = {
        "imgsz": train_params["imgsz"],
        "conf": params.get("predict_conf", 0.25),
        "iou": params.get("predict_iou", 0.7),
        "verbose": train_params.get("verbose", True),
    }
    predict_params = {**predict_defaults, **params.get("predict", {})}

    normalized = dict(params)
    normalized["model"] = model
    normalized["train"] = train_params
    normalized["val"] = val_params
    normalized["predict"] = predict_params
    normalized["export_test_predictions"] = params.get("export_test_predictions", True)
    return normalized


def main():
    args = parse_args()

    params_path = resolve_path(args.params, REPO_ROOT)
    params = normalize_params(load_yaml(params_path))

    data_path_value = args.data if args.data is not None else params["data_path"]
    data_config_path = resolve_existing_path(
        data_path_value,
        REPO_ROOT,
        params_path.parent,
    )
    data_config = load_yaml(data_config_path)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset_name = read_dataset_name(data_config)
    run_name = build_run_name(dataset_name, params, run_timestamp)
    run_dir = RUNS_DIR / run_name

    logger = setup_logger(run_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Parameter file: %s", params_path)
    logger.info("Data config file: %s", data_config_path)

    save_yaml(run_dir / "params_snapshot.yml", params)
    save_yaml(run_dir / "data_snapshot.yml", data_config)

    device = get_device()
    logger.info("Using device: %s", device)

    run_metadata = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "host": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params_path": str(params_path),
        "data_config_path": str(data_config_path),
        "dataset_name": dataset_name,
        "dataset": to_builtin(data_config),
        "training_params": to_builtin(params),
        "device": device,
    }

    metrics_summary = {}
    timings = {}
    artifacts = {
        "train_run_dir": str(run_dir),
        "weights_used_for_eval": None,
        "test_eval_dir": None,
        "test_predictions_dir": None,
        "submission_zip": None,
    }

    model_path = params.get("model", DEFAULT_MODEL)
    overall_start = time.perf_counter()
    logger.info("Loading pretrained YOLO model: %s", model_path)
    model = YOLO(model_path)

    train_start = time.perf_counter()
    train_kwargs = {
        **params["train"],
        "data": str(data_config_path),
        "device": device,
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": True,
    }
    train_results = model.train(**train_kwargs)
    actual_run_dir = Path(train_results.save_dir).resolve()
    if actual_run_dir != run_dir:
        logger.info("Ultralytics save_dir differs from planned run_dir: %s", actual_run_dir)
        run_dir = actual_run_dir
        run_metadata["run_dir"] = str(run_dir)
        run_metadata["run_name"] = run_dir.name
        artifacts["train_run_dir"] = str(run_dir)
    effective_run_name = run_dir.name
    timings["training_seconds"] = time.perf_counter() - train_start
    logger.info("Training finished in %.2f seconds", timings["training_seconds"])

    best_weights = run_dir / "weights" / "best.pt"
    final_weights = best_weights if best_weights.exists() else run_dir / "weights" / "last.pt"
    artifacts["weights_used_for_eval"] = str(final_weights)
    logger.info("Using weights for evaluation: %s", final_weights)

    best_validation_metrics = read_best_validation_metrics(run_dir)
    if best_validation_metrics is not None:
        logger.info("Best validation metrics: %s", json.dumps(best_validation_metrics, indent=2))

    eval_model = YOLO(str(final_weights))
    test_source = resolve_dataset_split(data_config_path, data_config, "test")
    logger.info("Resolved test split: %s", test_source)

    if split_has_labels(test_source):
        eval_start = time.perf_counter()
        test_eval_dir = RUNS_DIR / f"{effective_run_name}__test-eval"
        test_metrics = eval_model.val(
            data=str(data_config_path),
            split="test",
            device=device,
            project=str(RUNS_DIR),
            name=test_eval_dir.name,
            exist_ok=True,
            **params["val"],
        )
        timings["test_evaluation_seconds"] = time.perf_counter() - eval_start
        metrics_summary = extract_metrics(test_metrics)
        artifacts["test_eval_dir"] = str(test_eval_dir)
        logger.info("Test metrics: %s", json.dumps(metrics_summary, indent=2))
    else:
        metrics_summary = {
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50-95": None,
            "fitness": None,
            "raw": {},
            "status": "skipped",
            "reason": f"No label files found for test split at {test_source.parent / 'labels'}",
        }
        timings["test_evaluation_seconds"] = 0.0
        logger.warning("Skipping test metric evaluation: %s", metrics_summary["reason"])

    if params.get("export_test_predictions", True):
        predictions_dir = RUNS_DIR / f"{effective_run_name}__submission"
        predict_start = time.perf_counter()
        eval_model.predict(
            source=str(test_source),
            data=str(data_config_path),
            device=device,
            save=False,
            save_txt=True,
            save_conf=True,
            project=str(RUNS_DIR),
            name=predictions_dir.name,
            exist_ok=True,
            **params["predict"],
        )
        timings["test_prediction_seconds"] = time.perf_counter() - predict_start
        artifacts["test_predictions_dir"] = str(predictions_dir)

        submission_zip = zip_submission(predictions_dir)
        if submission_zip is not None:
            artifacts["submission_zip"] = str(submission_zip)
            logger.info("Submission zip written to: %s", submission_zip)

        logger.info("Saved YOLO-format test predictions in: %s", predictions_dir)
    else:
        timings["test_prediction_seconds"] = 0.0

    timings["total_seconds"] = time.perf_counter() - overall_start
    timings["total_hours"] = timings["total_seconds"] / 3600.0

    sustainability = {
        "combined_compute_time_seconds": timings["total_seconds"],
        "combined_compute_time_hours": timings["total_hours"],
        "note": (
            "Use this combined compute time for the sustainability section. "
            "If you later measure average GPU power on Cybele, you can estimate "
            "energy as power(W) * time(h) / 1000."
        ),
    }

    summary = {
        "run": run_metadata,
        "timings": timings,
        "sustainability": sustainability,
        "best_validation_metrics": best_validation_metrics,
        "test_metrics": metrics_summary,
        "artifacts": artifacts,
    }

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")

    registry_row = {
        "run_name": run_metadata["run_name"],
        "started_at": run_metadata["started_at"],
        "host": run_metadata["host"],
        "dataset": dataset_name,
        "model": Path(model_path).stem,
        "run_tag": params.get("run_tag", ""),
        "epochs": params["train"].get("epochs"),
        "batch_size": params["train"].get("batch"),
        "img_size": params["train"].get("imgsz"),
        "lr": params["train"].get("lr0"),
        "patience": params["train"].get("patience"),
        "optimizer": params["train"].get("optimizer"),
        "seed": params["train"].get("seed"),
        "predict_conf": params["predict"].get("conf"),
        "predict_iou": params["predict"].get("iou"),
        "device": device,
        "training_seconds": timings["training_seconds"],
        "test_evaluation_seconds": timings["test_evaluation_seconds"],
        "test_prediction_seconds": timings["test_prediction_seconds"],
        "total_hours": timings["total_hours"],
        "val_precision": None if best_validation_metrics is None else best_validation_metrics["precision"],
        "val_recall": None if best_validation_metrics is None else best_validation_metrics["recall"],
        "val_mAP50": None if best_validation_metrics is None else best_validation_metrics["mAP50"],
        "val_mAP50_95": None if best_validation_metrics is None else best_validation_metrics["mAP50-95"],
        "val_fitness": None if best_validation_metrics is None else best_validation_metrics["fitness"],
        "test_precision": metrics_summary.get("precision"),
        "test_recall": metrics_summary.get("recall"),
        "test_mAP50": metrics_summary.get("mAP50"),
        "test_mAP50_95": metrics_summary.get("mAP50-95"),
        "submission_zip": artifacts["submission_zip"],
        "params_json": json.dumps(to_builtin(params), sort_keys=True),
    }
    append_experiment_registry(EXPERIMENT_REGISTRY_PATH, registry_row)

    logger.info("Total runtime: %.2f seconds (%.4f hours)", timings["total_seconds"], timings["total_hours"])
    logger.info("Summary written to: %s", summary_path)
    logger.info("Experiment registry updated: %s", EXPERIMENT_REGISTRY_PATH)


if __name__ == "__main__":
    main()
