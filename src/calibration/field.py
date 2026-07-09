from pathlib import Path
import json

import cv2
import numpy as np


class FootballField:

    def __init__(
        self,
        config_path: Path,
        render_height_px: int | None = None,
    ):

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:

            config = json.load(f)

        # ==========================================
        # Medidas reales (metros)
        # ==========================================

        self.field_width_m = config["field_width_m"]
        self.field_height_m = config["field_height_m"]

        self.center_circle_radius_m = config["center_circle_radius_m"]

        self.penalty_area_width_m = config["penalty_area_width_m"]
        self.penalty_area_depth_m = config["penalty_area_depth_m"]

        # ==========================================
        # Configuración del render
        # ==========================================

        self.render_height_px = (
            render_height_px
            if render_height_px is not None
            else config["render_height_px"]
        )

        self.margin_px = config["margin_px"]

        aspect_ratio = self.field_width_m / self.field_height_m

        self.render_width_px = int(self.render_height_px * aspect_ratio)

        self.scale_x = self.render_width_px / self.field_width_m

        self.scale_y = self.render_height_px / self.field_height_m

    # ==================================================

    def load_image(
        self,
        image_path: Path,
    ):

        image = cv2.imdecode(
            np.fromfile(
                image_path,
                dtype=np.uint8,
            ),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise RuntimeError(f"No se pudo abrir {image_path}")

        h, w = image.shape[:2]

        scale = self.render_height_px / h

        image = cv2.resize(
            image,
            (
                int(w * scale),
                self.render_height_px,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return image

    # ==================================================

    def meters_to_pixels(
        self,
        x_m,
        y_m,
    ):

        x = int(x_m * self.scale_x)

        y = int(y_m * self.scale_y)

        return x, y

    # ==================================================

    def pixels_to_meters(
        self,
        x_px,
        y_px,
    ):

        x = x_px / self.scale_x

        y = y_px / self.scale_y

        return x, y
