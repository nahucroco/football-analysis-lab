from pathlib import Path

from ultralytics import YOLO


class YOLODetector:

    def __init__(
        self,
        model_path: Path,
        tracker: str = "bytetrack.yaml",
        confidence: float = 0.15,
        image_size: int = 1536,
    ):

        self.model = YOLO(str(model_path))

        self.tracker = tracker
        self.confidence = confidence
        self.image_size = image_size

    # ==================================================

    def detect(
        self,
        frame,
        classes=[0],
    ):

        return self.model.predict(
            source=frame,
            classes=classes,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

    # ==================================================

    def track(
        self,
        frame,
        classes=[0],
    ):

        return self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
