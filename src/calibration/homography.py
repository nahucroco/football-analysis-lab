from pathlib import Path
import json

import cv2
import numpy as np


class Homography:

    def __init__(
        self,
        calibration_path: Path,
    ):

        with open(
            calibration_path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        self.image_points = np.array(
            data["image_points"],
            dtype=np.float32,
        )

        self.field_points = np.array(
            data["field_points"],
            dtype=np.float32,
        )

        self.matrix = None

        self.inverse_matrix = None

    # ==================================================

    def compute(self):

        self.matrix, _ = cv2.findHomography(
            self.image_points,
            self.field_points,
        )

        self.inverse_matrix, _ = cv2.findHomography(
            self.field_points,
            self.image_points,
        )

    # ==================================================

    def transform(
        self,
        point,
    ):

        point = np.array(
            [[point]],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            point,
            self.matrix,
        )

        return tuple(transformed[0][0])

    # ==================================================

    def inverse_transform(
        self,
        point,
    ):

        point = np.array(
            [[point]],
            dtype=np.float32,
        )

        transformed = cv2.perspectiveTransform(
            point,
            self.inverse_matrix,
        )

        return tuple(transformed[0][0])
