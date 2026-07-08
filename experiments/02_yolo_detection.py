from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"

model = YOLO(str(MODEL_PATH))

print(model.names)