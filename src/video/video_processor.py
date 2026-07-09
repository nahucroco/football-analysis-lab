from pathlib import Path

import cv2


class VideoProcessor:

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
    ):

        self.cap = cv2.VideoCapture(str(input_path))

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir {input_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (
                self.width,
                self.height,
            ),
        )

    # ==========================================

    def read(self):

        return self.cap.read()

    # ==========================================

    def write(
        self,
        frame,
    ):

        self.writer.write(frame)

    # ==========================================

    def release(self):

        self.cap.release()

        self.writer.release()

        cv2.destroyAllWindows()
