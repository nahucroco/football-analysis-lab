from pathlib import Path

from ultralytics import YOLO


class BallDetector:

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.15,
        image_size: int = 1280,
    ):

        self.model = YOLO(str(model_path))

        self.confidence = confidence
        self.image_size = image_size

    # ==================================================

    def detect(
        self,
        frame,
    ):

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        return results

    # ==================================================

    def get_detections(
        self,
        frame,
    ):

        results = self.detect(frame)

        boxes = results[0].boxes

        print(len(boxes))

        detections = []

        for box in boxes:

            print(f"Ball conf: {float(box.conf):.3f}")

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0],
            )

            detections.append(
                {
                    "center": (
                        (x1 + x2) // 2,
                        (y1 + y2) // 2,
                    ),
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "confidence": float(box.conf),
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )

        return detections
