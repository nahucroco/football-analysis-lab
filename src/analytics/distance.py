import math


class DistanceTracker:

    def __init__(
        self,
        field,
    ):

        self.field = field

        self.positions = {}

        self.distances = {}

    # ==================================================

    def update(
        self,
        track_id,
        position,
    ):

        if track_id not in self.positions:

            self.positions[track_id] = position
            self.distances[track_id] = 0.0

            return

        previous = self.positions[track_id]

        dx = position[0] - previous[0]

        dy = position[1] - previous[1]

        dx_m = dx / self.field.scale_x
        dy_m = dy / self.field.scale_y

        distance = math.sqrt(dx_m**2 + dy_m**2)

        self.distances[track_id] += distance

        self.positions[track_id] = position

    # ==================================================

    def get_distance(
        self,
        track_id,
    ):

        return self.distances.get(
            track_id,
            0.0,
        )

    # ==================================================

    def get_all_distances(self):

        return self.distances

    # ==================================================

    def reset(self):

        self.positions.clear()

        self.distances.clear()
