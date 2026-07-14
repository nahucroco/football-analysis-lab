from pathlib import Path

from ultralytics import YOLO


class PlayerDetector:

    def __init__(
        self,
        model_path: Path,
        tracker="bytetrack.yaml",
        player_confidence=0.57,
        default_confidence=0.25,
        image_size=1280,
    ):

        self.model = YOLO(str(model_path))

        self.tracker = tracker
        self.player_confidence = player_confidence
        self.default_confidence = default_confidence
        self.image_size = image_size

    # ==================================================

    def detect(
        self,
        frame,
    ):

        return self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=0.25,
            verbose=False,
        )

    # ==================================================

    def track(
        self,
        frame,
        classes=[0, 1, 2],
    ):

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            classes=classes,
            conf=self.default_confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        boxes = results[0].boxes

        player_class = next(
            cls for cls, name in results[0].names.items() if name == "Player"
        )

        keep = []

        for i, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):

            if int(cls) == player_class and float(conf) < self.player_confidence:
                continue

            keep.append(i)

        results[0].boxes = boxes[keep]

        return results
