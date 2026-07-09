class MovementTracker:

    STOPPED = "STOPPED"
    WALKING = "WALKING"
    JOGGING = "JOGGING"
    RUNNING = "RUNNING"
    SPRINT = "SPRINT"

    def __init__(
        self,
        fps,
        walking_speed=7.0,
        jogging_speed=15.0,
        running_speed=22.0,
    ):

        self.fps = fps

        self.thresholds = {
            self.STOPPED: 1.0,
            self.WALKING: 7.0,
            self.JOGGING: 15.0,
            self.RUNNING: 22.0,
        }

        self.current_state = {}

        self.frames = {}

    # ==================================================

    def update(
        self,
        track_id,
        speed,
    ):

        if track_id not in self.frames:

            self.frames[track_id] = {
                self.STOPPED: 0,
                self.WALKING: 0,
                self.JOGGING: 0,
                self.RUNNING: 0,
                self.SPRINT: 0,
            }

            self.current_state[track_id] = self.STOPPED

        state = self._classify(speed)

        self.current_state[track_id] = state

        self.frames[track_id][state] += 1

    # ==================================================

    def _classify(
        self,
        speed,
    ):

        if speed < self.thresholds[self.STOPPED]:
            return self.STOPPED

        if speed < self.thresholds[self.WALKING]:
            return self.WALKING

        if speed < self.thresholds[self.JOGGING]:
            return self.JOGGING

        if speed < self.thresholds[self.RUNNING]:
            return self.RUNNING

        return self.SPRINT

    # ==================================================

    def get_state(
        self,
        track_id,
    ):

        return self.current_state.get(
            track_id,
            self.STOPPED,
        )

    # ==================================================

    def get_time(
        self,
        track_id,
        state,
    ):

        if track_id not in self.frames:

            return 0.0

        return self.frames[track_id][state] / self.fps

    # ==================================================

    def get_statistics(
        self,
        track_id,
    ):

        if track_id not in self.frames:

            return None

        total_frames = sum(self.frames[track_id].values())

        if total_frames == 0:

            return None

        statistics = {}

        for state, frames in self.frames[track_id].items():

            statistics[state] = {
                "frames": frames,
                "seconds": frames / self.fps,
                "percentage": (frames / total_frames) * 100,
            }

        return statistics

    # ==================================================

    def get_all_statistics(self):

        return {track_id: self.get_statistics(track_id) for track_id in self.frames}

    # ==================================================

    def reset(self):

        self.current_state.clear()

        self.frames.clear()
