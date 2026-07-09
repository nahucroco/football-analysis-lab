import math
from collections import defaultdict, deque


class SpeedTracker:

    DEFAULT_WINDOW = 10
    DEFAULT_ALPHA = 0.3

    def __init__(
        self,
        field,
        fps,
        window_size=DEFAULT_WINDOW,
        alpha=DEFAULT_ALPHA,
    ):

        self.field = field

        self.fps = fps

        self.position_history = defaultdict(lambda: deque(maxlen=self.window_size))

        self.current_speed = {}

        self.max_speed = {}

        self.total_speed = {}

        self.speed_samples = {}

        self.window_size = window_size

        self.alpha = alpha

    # ==================================================

    def update(
        self,
        track_id,
        position,
    ):

        if track_id not in self.current_speed:

            self.current_speed[track_id] = 0.0
            self.max_speed[track_id] = 0.0
            self.total_speed[track_id] = 0.0
            self.speed_samples[track_id] = 0

        history = self.position_history[track_id]

        history.append(position)

        if len(history) < self.window_size:
            return

        speed_kmh = self._calculate_speed(
            track_id,
            history,
        )

        self.current_speed[track_id] = speed_kmh

        if speed_kmh > self.max_speed[track_id]:

            self.max_speed[track_id] = speed_kmh

        self.total_speed[track_id] += speed_kmh

        self.speed_samples[track_id] += 1

    # ==================================================

    def get_speed(
        self,
        track_id,
    ):

        return self.current_speed.get(
            track_id,
            0.0,
        )

    # ==================================================

    def get_average_speed(
        self,
        track_id,
    ):

        samples = self.speed_samples.get(
            track_id,
            0,
        )

        if samples == 0:

            return 0.0

        return self.total_speed[track_id] / samples

    # ==================================================

    def get_max_speed(
        self,
        track_id,
    ):

        return self.max_speed.get(
            track_id,
            0.0,
        )

    # ==================================================

    def get_all_max_speeds(self):

        return self.max_speed

    # ==================================================

    def reset(self):

        self.position_history.clear()

        self.current_speed.clear()

        self.max_speed.clear()

        self.total_speed.clear()

        self.speed_samples.clear()

    # ==================================================

    def _calculate_speed(
        self,
        track_id,
        history,
    ):

        old = history[0]
        new = history[-1]

        dx = new[0] - old[0]
        dy = new[1] - old[1]

        dx_m = dx / self.field.scale_x
        dy_m = dy / self.field.scale_y

        distance_m = math.sqrt(dx_m**2 + dy_m**2)

        elapsed_time = (len(history) - 1) / self.fps

        speed_ms = distance_m / elapsed_time

        speed_kmh = speed_ms * 3.6

        speed_kmh = (
            self.alpha * speed_kmh + (1 - self.alpha) * self.current_speed[track_id]
        )

        return speed_kmh

    # ==================================================

    def _pixels_to_meters(
        self,
        dx,
        dy,
    ):

        dx_m = dx / self.field.scale_x
        dy_m = dy / self.field.scale_y

        return math.sqrt(dx_m**2 + dy_m**2)
