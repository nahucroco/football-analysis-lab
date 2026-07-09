from pathlib import Path
import time

import cv2

from src.video.video_processor import VideoProcessor
from src.detection.yolo_detector import YOLODetector
from src.visualization.renderer import Renderer
from src.calibration.field import FootballField
from src.calibration.homography import Homography

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"

OUTPUT_PATH = BASE_DIR / "outputs" / "partido_top_view.mp4"

MODEL_PATH = BASE_DIR / "models" / "yolo11m.pt"

FIELD_CONFIG_PATH = BASE_DIR / "config" / "field_5v5.json"

FIELD_IMAGE_PATH = BASE_DIR / "assets" / "fields" / "football5.png"

CALIBRATION_PATH = BASE_DIR / "config" / "camera_calibration.json"

DISPLAY_SCALE = 0.50


# ==========================================================
# Inicialización
# ==========================================================

processor = VideoProcessor(
    VIDEO_PATH,
    OUTPUT_PATH,
)

detector = YOLODetector(
    MODEL_PATH,
    tracker="bytetrack.yaml",
)

renderer = Renderer()

field = FootballField(
    FIELD_CONFIG_PATH,
    render_height_px=processor.height,
)

field_image = field.load_image(
    FIELD_IMAGE_PATH,
)

homography = Homography(
    CALIBRATION_PATH,
)

homography.compute()

canvas_width = processor.width + field_image.shape[1] + 20

canvas_height = processor.height

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    processor.fps,
    (
        canvas_width,
        canvas_height,
    ),
)

print(f"Resolución: {processor.width}x{processor.height}")

print(f"FPS: {processor.fps:.2f}")

frame_number = 0

start = time.perf_counter()

# ==========================================================
# Loop principal
# ==========================================================

while True:

    ret, frame = processor.read()

    if not ret:
        break

    frame_number += 1

    print(
        f"\rProcesando frame {frame_number}/{processor.total_frames}",
        end="",
    )

    results = detector.track(frame)

    boxes = results[0].boxes

    video_frame = frame.copy()

    field_frame = field_image.copy()

    renderer.draw_title(
        video_frame,
        "VIDEO",
    )

    renderer.draw_title(
        field_frame,
        "TOP VIEW",
    )

    renderer.draw_boxes(
        video_frame,
        boxes,
    )

    # ==========================================
    # Transformar cada jugador
    # ==========================================

    for box in boxes:

        if box.id is None:
            continue

        track_id = int(box.id)

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0],
        )

        # Punto de contacto con el suelo

        cx = (x1 + x2) // 2
        cy = y2

        mapped = homography.transform((cx, cy))

        renderer.draw_player(
            field_frame,
            mapped,
            track_id,
        )

    # ==========================================
    # Mostrar información
    # ==========================================

    renderer.draw_text(
        video_frame,
        f"Frame: {frame_number}",
        (20, 70),
    )

    renderer.draw_text(
        video_frame,
        f"Jugadores: {len(boxes)}",
        (20, 105),
    )

    renderer.draw_text(
        field_frame,
        f"Jugadores: {len(boxes)}",
        (20, 70),
    )

    # ==========================================
    # Unir vistas
    # ==========================================

    canvas = renderer.stack_views(
        video_frame,
        field_frame,
    )

    print(canvas.shape)

    display = cv2.resize(
        canvas,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    cv2.imshow(
        "Football Analysis Lab - Top View",
        display,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    writer.write(canvas)


# ==========================================================
# Finalización
# ==========================================================

writer.release()

processor.release()

elapsed = time.perf_counter() - start

print()

print("=" * 50)
print("TOP VIEW REPORT")
print("=" * 50)

print(f"Modelo: {MODEL_PATH.stem}")
print(f"Tracker: {detector.tracker}")

print(f"Frames procesados: {frame_number}")

print(f"Tiempo total: {elapsed:.2f} s")
print(f"Tiempo por frame: {elapsed/frame_number:.4f} s")
print(f"FPS efectivos: {frame_number/elapsed:.2f}")

print("Video generado correctamente.")
