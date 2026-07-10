import math


class BallTracker:

    def __init__(
        self,
        max_size=40,
        min_confidence=0.20,
        max_distance=120,
        max_missing_frames=5,
    ):

        self.max_size = max_size
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames

        self.position = None

        self.bbox = None

        self.confidence = 0.0

        self.missing_frames = 0

    # ==================================================

    def update(
        self,
        detections,
    ):

        detection = self._select_detection(
            detections,
        )

        if detection is None:

            self.missing_frames += 1

            if self.missing_frames > self.max_missing_frames:

                self.position = None
                self.bbox = None
                self.confidence = 0.0

            return self.position

        self.position = detection["center"]

        self.bbox = detection["bbox"]

        self.confidence = detection["confidence"]

        self.missing_frames = 0

        return self.position

    # ==================================================

    def _select_detection(
        self,
        detections,
    ):

        candidates = []

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            width = x2 - x1
            height = y2 - y1

            if width > self.max_size:
                continue

            if height > self.max_size:
                continue

            ratio = width / max(height, 1)

            if ratio < 0.6 or ratio > 1.4:
                continue

            if detection["confidence"] < self.min_confidence:
                continue

            if self.position is not None:

                cx, cy = detection["center"]

                distance = math.hypot(
                    cx - self.position[0],
                    cy - self.position[1],
                )

                if distance > self.max_distance:
                    continue

            candidates.append(detection)

        if not candidates:
            return None

        candidates.sort(
            key=lambda d: d["confidence"],
            reverse=True,
        )

        return candidates[0]

    # ==================================================

    def get_position(self):

        return self.position

    # ==================================================

    def get_bbox(self):

        return self.bbox

    # ==================================================

    def is_visible(self):

        return self.position is not None

    # ==================================================

    def reset(self):

        self.position = None

        self.bbox = None

        self.confidence = 0.0

        self.missing_frames = 0
