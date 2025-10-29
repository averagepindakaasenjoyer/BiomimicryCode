import cv2
import numpy as np
from ultralytics import YOLO
import os
import dotenv

dotenv.load_dotenv()

model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=10, imgsz=640)

print("Training completed.")

# test the model

results = model.val()
print("Validation completed.")
for result in results:
    print(result.metrics)



