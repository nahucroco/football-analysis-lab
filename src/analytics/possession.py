import math


class PossessionTracker:

    def __init__(
        self,
        fps,
        max_distance=35,
        confirmation_frames=5,
    ):

        self.fps = fps

        self.max_distance = max_distance

        self.confirmation_frames = confirmation_frames

        self.current_owner = None

        self.candidate_owner = None

        self.candidate_frames = 0

        self.possession_frames = {}

    # ==================================================

    def update(
        self,
        players,
        ball_position,
    ):

        if ball_position is None:

            return

        owner = self._nearest_player(
            players,
            ball_position,
        )

        if owner is None:

            return

        if owner == self.current_owner:

            self.candidate_owner = None
            self.candidate_frames = 0

        else:

            if owner == self.candidate_owner:

                self.candidate_frames += 1

            else:

                self.candidate_owner = owner
                self.candidate_frames = 1

            if self.candidate_frames >= self.confirmation_frames:

                self.current_owner = owner

                self.candidate_owner = None

                self.candidate_frames = 0

        if self.current_owner is not None:

            self.possession_frames.setdefault(
                self.current_owner,
                0,
            )

            self.possession_frames[self.current_owner] += 1

    # ==================================================

    def _nearest_player(
        self,
        players,
        ball,
    ):

        nearest = None

        nearest_distance = float("inf")

        bx, by = ball

        for track_id, position in players.items():

            px, py = position

            distance = math.hypot(
                px - bx,
                py - by,
            )

            if distance < nearest_distance and distance <= self.max_distance:

                nearest_distance = distance

                nearest = track_id

        return nearest

    # ==================================================

    def get_owner(self):

        return self.current_owner

    # ==================================================

    def get_possession_time(
        self,
        track_id,
    ):

        frames = self.possession_frames.get(
            track_id,
            0,
        )

        return frames / self.fps

    # ==================================================

    def get_all_possession_times(self):

        return {
            track_id: frames / self.fps
            for track_id, frames in self.possession_frames.items()
        }

    # ==================================================

    def reset(self):

        self.current_owner = None

        self.candidate_owner = None

        self.candidate_frames = 0

        self.possession_frames.clear()
