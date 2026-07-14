import math
from enum import Enum


class BallState(Enum):

    UNINITIALIZED = 0

    TRACKED = 1

    LOST = 2

    RECOVERING = 3


class BallTracker:

    def __init__(
        self,
        max_jump_distance=250,
        max_missing_frames=8,
    ):

        self.position = None

        self.previous_position = None

        self.velocity = (0.0, 0.0)

        self.last_detection = None

        self.bbox = None

        self.state = BallState.UNINITIALIZED

        self.missing_frames = 0

        self.max_missing_frames = max_missing_frames

        self.max_jump_distance = max_jump_distance

        self.prediction_decay = 0.90

    # ==================================================

    def _distance(
        self,
        p1,
        p2,
    ):

        return math.hypot(
            p1[0] - p2[0],
            p1[1] - p2[1],
        )

    # ==================================================

    def _update_velocity(self):

        if self.previous_position is None or self.position is None:

            self.velocity = (0.0, 0.0)

            return

        dx = self.position[0] - self.previous_position[0]

        dy = self.position[1] - self.previous_position[1]

        self.velocity = (
            dx,
            dy,
        )

    # ==================================================

    def _select_detection(
        self,
        detections,
    ):

        if not detections:
            return None

        if self.position is None:

            return max(
                detections,
                key=lambda d: d["confidence"],
            )

        return min(
            detections,
            key=lambda d: self._distance(
                d["center"],
                self.position,
            ),
        )

    # ==================================================

    def update(
        self,
        detections,
    ):

        print(self.state)

        if self.state == BallState.UNINITIALIZED:

            self._update_uninitialized(
                detections,
            )

        elif self.state == BallState.TRACKED:

            self._update_tracked(
                detections,
            )

        elif self.state == BallState.LOST:

            self._update_lost(
                detections,
            )

        elif self.state == BallState.RECOVERING:

            self._update_recovering(
                detections,
            )

        """selected = self._select_detection(
            detections,
        )

        if selected is None:

            self.missing_frames += 1

            if self.missing_frames <= self.max_missing_frames:

                self._predict_position()

            else:

                self.reset()

            return

        self.missing_frames = 0

        self.previous_position = self.position

        self.position = selected["center"]

        self.last_detection = selected

        self.missing_frames = 0

        self.last_detection = selected

        self.bbox = selected["bbox"]

        self._update_velocity()"""

    # ==================================================

    def get_position(self):

        return self.position

    # ==================================================

    def get_velocity(self):

        return self.velocity

    # ==================================================

    def get_bbox(self):

        return self.bbox

    # ==================================================

    def is_visible(self):

        return self.position is not None

    # ==================================================

    def reset(self):

        self.position = None

        self.previous_position = None

        self.velocity = (0.0, 0.0)

        self.last_detection = None

        self.bbox = None

        self.missing_frames = 0

    # ==================================================

    def _predict_position(self):

        if self.position is None:

            return

        vx = self.velocity[0] * self.prediction_decay
        vy = self.velocity[1] * self.prediction_decay

        self.velocity = (
            vx,
            vy,
        )

        self.position = (
            self.position[0] + vx,
            self.position[1] + vy,
        )

    # ==================================================

    def _update_uninitialized(
        self,
        detections,
    ):

        selected = self._select_detection(
            detections,
        )

        if selected is None:
            return

        self.position = selected["center"]

        self.previous_position = None

        self.velocity = (0.0, 0.0)

        self.last_detection = selected

        self.bbox = selected["bbox"]

        self.missing_frames = 0

        self.state = BallState.TRACKED

    # ==================================================

    def _update_tracked(
        self,
        detections,
    ):

        print(f"detecciones: {len(detections)}")

        selected = self._select_detection(
            detections,
        )

        if selected is None:
            print("NO HAY DETECCIONES")

            self.state = BallState.LOST
            self.missing_frames = 1
            return

        distance = self._distance(
            selected["center"],
            self.position,
        )

        print(
            "Nueva:",
            selected["center"],
            "Actual:",
            self.position,
        )

        if distance > self.max_jump_distance:

            self.state = BallState.LOST
            self.missing_frames = 1
            return

        print(f"TRACKED -> {selected['center']}")

        self.previous_position = self.position

        self.position = selected["center"]

        self.last_detection = selected

        self.bbox = selected["bbox"]

        self.missing_frames = 0

        self._update_velocity()

    # ==================================================

    def _update_lost(
        self,
        detections,
    ):

        pass

    # ==================================================

    def _update_recovering(
        self,
        detections,
    ):

        pass
