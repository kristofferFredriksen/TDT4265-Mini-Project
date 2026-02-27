from ultralytics import YOLO
import torch as t
import yaml
import time
import logging
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr


# Create directory for saving training results with a unique timestamp
run_timestamp = time.strftime("%Y%m%d-%H%M%S")
run_name = f"yolo_26n_{run_timestamp}"
run_dir = Path(f"runs/{run_name}")
run_dir.mkdir(parents=True, exist_ok=False)

# Python logging setup to log training progress and results

log_file = run_dir / "python.log"

logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.info(f"Run dir: {run_dir}")

# Load training parameters from YAML file    
with open('config/yolo_params.yml', 'r') as file:
    params = yaml.safe_load(file)

# Set device for training (GPU if available, otherwise CPU)
device = "0" if t.cuda.is_available() and t.cuda.device_count() > 0 else "cpu"
print("Using device:", device)

# Loading pretrained YOLO on COCO dataset
print("Loading pretrained YOLO model...")
model = YOLO("yolo26n.pt")
print("Model loaded successfully.")

# Redirecting stdout and stderr to a log file to capture training output
console_log_path = run_dir / "console.log"
with open(console_log_path, "w") as f, redirect_stdout(f), redirect_stderr(f):
    results = model.train(data = "./config/data_roadpoles_v1.yaml", 
                          epochs=params['epochs'], 
                          batch=params['batch_size'], 
                          imgsz=params['img_size'], 
                          device=device, 
                          verbose=params['verbose'],
                          # Save results to the run directory (weights/plots/results.txt etc.)
                          project=run_dir.parent,
                          name=run_name,
                          exist_ok=False)

print("Training completed. Results saved to:", run_dir)
logger.info(f"Training completed. Results saved to: {run_dir}")

