import argparse
import json
import logging
import socket
import time
from pathlib import Path

import torch as t
import yaml
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = REPO_ROOT / "config" / "yolo_params.yml"
DEFAULT_MODEL = "yolo26n.pt"


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


def load_yaml(path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logger(run_dir):
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


def resolve_dataset_split(data_config_path, data_config, split_name):
    dataset_root = resolve_path(data_config["path"], data_config_path.parent)
    split_path = Path(data_config[split_name])
    if split_path.is_absolute():
        return split_path
    return (dataset_root / split_path).resolve()


def main():
    args = parse_args()

    params_path = resolve_path(args.params, REPO_ROOT)
    params = load_yaml(params_path)

    data_path_value = args.data if args.data is not None else params["data_path"]
    data_config_path = resolve_path(data_path_value, params_path.parent)
    data_config = load_yaml(data_config_path)

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    dataset_name = read_dataset_name(data_config)
    run_name = f"yolo_26n_{dataset_name}_{run_timestamp}"
    run_dir = REPO_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = setup_logger(run_dir)
    logger.info("Run directory: %s", run_dir)
    logger.info("Parameter file: %s", params_path)
    logger.info("Data config file: %s", data_config_path)

    device = get_device()
    logger.info("Using device: %s", device)

    run_metadata = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "host": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params_path": str(params_path),
        "data_config_path": str(data_config_path),
        "dataset": to_builtin(data_config),
        "training_params": to_builtin(params),
        "device": device,
    }

    metrics_summary = {}
    timings = {}

    overall_start = time.perf_counter()
    logger.info("Loading pretrained YOLO model: %s", DEFAULT_MODEL)
    model = YOLO(DEFAULT_MODEL)

    train_start = time.perf_counter()
    train_kwargs = {
        "data": str(data_config_path),
        "epochs": params["epochs"],
        "batch": params["batch_size"],
        "imgsz": params["img_size"],
        "device": device,
        "verbose": params.get("verbose", True),
        "project": str(REPO_ROOT / "runs"),
        "name": run_name,
        "exist_ok": False,
    }
    if params.get("lr") is not None:
        train_kwargs["lr0"] = params["lr"]
    model.train(**train_kwargs)
    timings["training_seconds"] = time.perf_counter() - train_start
    logger.info("Training finished in %.2f seconds", timings["training_seconds"])

    best_weights = run_dir / "weights" / "best.pt"
    final_weights = best_weights if best_weights.exists() else run_dir / "weights" / "last.pt"
    logger.info("Using weights for evaluation: %s", final_weights)

    eval_model = YOLO(str(final_weights))
    eval_start = time.perf_counter()
    test_metrics = eval_model.val(
        data=str(data_config_path),
        split="test",
        imgsz=params["img_size"],
        batch=params["batch_size"],
        device=device,
        verbose=params.get("verbose", True),
        project=str(REPO_ROOT / "runs"),
        name=f"{run_name}_test_eval",
        exist_ok=True,
    )
    timings["test_evaluation_seconds"] = time.perf_counter() - eval_start
    metrics_summary = extract_metrics(test_metrics)
    logger.info("Test metrics: %s", json.dumps(metrics_summary, indent=2))

    if params.get("export_test_predictions", True):
        test_source = resolve_dataset_split(data_config_path, data_config, "test")
        logger.info("Resolved test split for prediction export: %s", test_source)
        predict_start = time.perf_counter()
        eval_model.predict(
            source=str(test_source),
            data=str(data_config_path),
            imgsz=params["img_size"],
            device=device,
            save=False,
            save_txt=True,
            save_conf=True,
            project=str(REPO_ROOT / "runs"),
            name=f"{run_name}_test_predictions",
            exist_ok=True,
            verbose=params.get("verbose", True),
        )
        timings["test_prediction_seconds"] = time.perf_counter() - predict_start
        logger.info(
            "Saved YOLO-format test predictions with confidences in runs/%s_test_predictions",
            run_name,
        )

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
        "test_metrics": metrics_summary,
        "artifacts": {
            "train_run_dir": str(run_dir),
            "weights_used_for_eval": str(final_weights),
            "test_eval_dir": str(REPO_ROOT / "runs" / f"{run_name}_test_eval"),
            "test_predictions_dir": (
                str(REPO_ROOT / "runs" / f"{run_name}_test_predictions")
                if params.get("export_test_predictions", True)
                else None
            ),
        },
    }

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")

    logger.info("Total runtime: %.2f seconds (%.4f hours)", timings["total_seconds"], timings["total_hours"])
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
