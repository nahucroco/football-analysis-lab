import numpy as np


class HeatmapTracker:

    def __init__(
        self,
        field,
        rows=25,
        cols=40,
    ):

        self.field = field

        self.rows = rows
        self.cols = cols

        self.cell_width = field.render_width_px / cols
        self.cell_height = field.render_height_px / rows

        self.heatmaps = {}

    # ==================================================

    def update(
        self,
        track_id,
        position,
    ):

        if track_id not in self.heatmaps:

            self.heatmaps[track_id] = np.zeros(
                (
                    self.rows,
                    self.cols,
                ),
                dtype=np.int32,
            )

        x, y = position

        col = int(x / self.cell_width)
        row = int(y / self.cell_height)

        col = max(
            0,
            min(
                col,
                self.cols - 1,
            ),
        )

        row = max(
            0,
            min(
                row,
                self.rows - 1,
            ),
        )

        self.heatmaps[track_id][row, col] += 1

    # ==================================================

    def get_heatmap(
        self,
        track_id,
    ):

        return self.heatmaps.get(
            track_id,
            None,
        )

    # ==================================================

    def get_all_heatmaps(self):

        return self.heatmaps

    # ==================================================

    def reset(self):

        self.heatmaps.clear()
