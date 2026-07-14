from pathlib import Path
import time

import cv2

from src.video.video_processor import VideoProcessor
from src.detection.player_detector import YOLODetector
from src.visualization.renderer import Renderer
from src.calibration.field import FootballField
from src.calibration.homography import Homography
from src.analytics.distance import DistanceTracker
from src.analytics.speed import SpeedTracker

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"

OUTPUT_PATH = BASE_DIR / "outputs" / "partido_speed.mp4"

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

distance_tracker = DistanceTracker(
    field,
)

speed_tracker = SpeedTracker(
    field,
    processor.fps,
)

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

top_tracks = {}

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

    # ==========================================
    # DEBUG: mostrar clases detectadas
    # ==========================================

    for box in boxes:

        cls = int(box.cls)

        print(detector.model.names[cls])

    video_frame = frame.copy()

    field_frame = field_image.copy()

    renderer.draw_title(
        video_frame,
        "VIDEO",
    )

    renderer.draw_title(
        field_frame,
        "TOP VIEW TRAJECTORIES",
    )

    renderer.draw_boxes(
        video_frame,
        boxes,
    )

    # ==========================================
    # Transformar cada jugador
    # ==========================================

    renderer.draw_top_trajectories(
        field_frame,
        top_tracks,
    )

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

        distance_tracker.update(
            track_id,
            mapped,
        )

        speed_tracker.update(
            track_id,
            mapped,
        )

        top_tracks.setdefault(track_id, []).append(mapped)

        renderer.draw_player(
            field_frame,
            mapped,
            track_id,
        )

        distance = distance_tracker.get_distance(
            track_id,
        )

        speed = speed_tracker.get_speed(
            track_id,
        )

        renderer.draw_text(
            field_frame,
            f"{distance:.1f} m",
            (
                int(mapped[0]) + 15,
                int(mapped[1]) + 8,
            ),
            scale=0.45,
        )

        renderer.draw_text(
            field_frame,
            f"{speed:.1f} km/h",
            (
                int(mapped[0]) + 15,
                int(mapped[1]) + 24,
            ),
            scale=0.45,
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

print("=" * 60)
print("SPEED REPORT")
print("=" * 60)

distances = distance_tracker.get_all_distances()

max_speeds = speed_tracker.get_all_max_speeds()

for track_id in sorted(
    distances,
    key=distances.get,
    reverse=True,
):

    average_speed = speed_tracker.get_average_speed(
        track_id,
    )

    print(
        f"Jugador {track_id:>3} | "
        f"{distances[track_id]:6.2f} m | "
        f"Prom: {average_speed:5.2f} km/h | "
        f"Max: {max_speeds[track_id]:5.2f} km/h"
    )

print(f"Modelo: {MODEL_PATH.stem}")
print(f"Tracker: {detector.tracker}")

print(f"Frames procesados: {frame_number}")

print(f"Tiempo total: {elapsed:.2f} s")
print(f"Tiempo por frame: {elapsed/frame_number:.4f} s")
print(f"FPS efectivos: {frame_number/elapsed:.2f}")

print("Video generado correctamente.")
