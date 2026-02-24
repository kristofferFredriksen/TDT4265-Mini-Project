from ultralytics import YOLO
import torch as t

device = "0" if t.cuda.is_available() and t.cuda.device_count() > 0 else "cpu"
print("Using device:", device)

#Loading pretrained YOLO on COCO dataset
print("Loading pretrained YOLO model...")
model = YOLO("yolo26n.pt")
print("Model loaded successfully.")

print ("Starting training...")
results = model.train(data = "./src/data_roadpoles_v1.yaml", epochs=20, batch = 16, imgsz = 640, device="device", verbose = True)
print("Training completed.")
