from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEOS_DIR = BASE_DIR / "videos"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.25
IOU = 0.45
IMAGE_SIZE = 960

CLASSES = [0]