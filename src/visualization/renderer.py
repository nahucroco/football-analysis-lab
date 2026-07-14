from turtle import color

import cv2
import numpy as np


class Renderer:

    def draw_boxes(
        self,
        frame,
        boxes,
    ):

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id)

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0],
            )

            color = self.get_color(track_id)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        return frame

    # ==================================================

    def draw_trajectories(
        self,
        frame,
        tracks,
        history=40,
    ):

        for track_id, positions in tracks.items():

            if len(positions) < 2:
                continue

            color = self.get_color(track_id)

            positions = positions[-history:]

            for i in range(1, len(positions)):

                p1 = positions[i - 1]["center"]

                p2 = positions[i]["center"]

                cv2.line(
                    frame,
                    p1,
                    p2,
                    color,
                    2,
                )

        return frame

    # ==================================================

    def draw_text(
        self,
        frame,
        text,
        position,
        color=(255, 255, 255),
        scale=0.8,
    ):

        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
        )

        return frame

    # ==================================================

    def draw_circle(
        self,
        frame,
        center,
        color=(0, 255, 255),
        radius=4,
    ):

        cv2.circle(
            frame,
            center,
            radius,
            color,
            -1,
        )

        return frame

    # ==================================================

    def draw_player(
        self,
        frame,
        position,
        track_id,
        radius=8,
    ):

        x = int(position[0])
        y = int(position[1])

        color = self.get_color(track_id)

        cv2.circle(
            frame,
            (x, y),
            radius,
            color,
            -1,
        )

        cv2.circle(
            frame,
            (x, y),
            radius,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            str(track_id),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        return frame

    # ==================================================

    def get_color(
        self,
        track_id,
    ):

        return (
            (track_id * 37) % 255,
            (track_id * 91) % 255,
            (track_id * 53) % 255,
        )

        # ==================================================

    # ==================================================

    def stack_views(
        self,
        left,
        right,
        separator=20,
    ):

        sep = np.full(
            (
                left.shape[0],
                separator,
                3,
            ),
            40,
            dtype=np.uint8,
        )

        return np.hstack(
            (
                left,
                sep,
                right,
            )
        )

        # ==================================================

    # ==================================================

    def draw_title(
        self,
        frame,
        title,
    ):

        cv2.putText(
            frame,
            title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return frame

    # ==================================================

    def draw_top_trajectories(
        self,
        frame,
        tracks,
        history=80,
    ):

        for track_id, positions in tracks.items():

            if len(positions) < 2:
                continue

            color = self.get_color(track_id)

            positions = positions[-history:]

            for i in range(1, len(positions)):

                p1 = (
                    int(positions[i - 1][0]),
                    int(positions[i - 1][1]),
                )

                p2 = (
                    int(positions[i][0]),
                    int(positions[i][1]),
                )

                cv2.line(
                    frame,
                    p1,
                    p2,
                    color,
                    2,
                )

        return frame

    # ==================================================

    def draw_ball(
        self,
        frame,
        position,
        radius=8,
        color=(0, 0, 255),
        label=True,
    ):

        if position is None:
            return

        x, y = map(int, position)

        cv2.circle(
            frame,
            (x, y),
            radius,
            color,
            -1,
        )

        cv2.circle(
            frame,
            (x, y),
            radius + 2,
            (255, 255, 255),
            2,
        )

        if label:

            cv2.putText(
                frame,
                "BALL",
                (x + 12, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
