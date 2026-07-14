from pathlib import Path
import time

import cv2

from src.video.video_processor import VideoProcessor
from src.detection.player_detector import PlayerDetector
from src.detection.ball_detector import BallDetector
from src.visualization.renderer import Renderer
from src.calibration.field import FootballField
from src.calibration.homography import Homography
from src.analytics.distance import DistanceTracker
from src.analytics.speed import SpeedTracker

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"

OUTPUT_PATH = BASE_DIR / "outputs" / "partido_ball_tracker.mp4"

PLAYER_MODEL_PATH = BASE_DIR / "models" / "football" / "best.pt"

BALL_MODEL_PATH = BASE_DIR / "models" / "soccer_ball" / "v9b_best.pt"

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

player_detector = PlayerDetector(
    PLAYER_MODEL_PATH,
)

ball_detector = BallDetector(
    BALL_MODEL_PATH,
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

top_tracks = {}

start = time.perf_counter()

# ==========================================================
# Loop principal
# ==========================================================

ball_frames = 0
ball_missed_frames = 0

total_ball_confidence = 0.0
total_ball_detections = 0

while True:

    ret, frame = processor.read()

    if not ret:
        break

    frame_number += 1

    print(
        f"\rProcesando frame {frame_number}/{processor.total_frames}",
        end="",
    )

    player_results = player_detector.detect(frame)
    player_boxes = player_results[0].boxes

    ball_detections = ball_detector.get_detections(frame)

    if ball_detections:

        ball_frames += 1

        for detection in ball_detections:

            total_ball_confidence += detection["confidence"]
            total_ball_detections += 1

    else:

        ball_missed_frames += 1

    # ==========================================
    # DEBUG: mostrar clases detectadas
    # ==========================================

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
        player_boxes,
    )

    # ==========================================
    # Transformar cada jugador
    # ==========================================

    renderer.draw_top_trajectories(
        field_frame,
        top_tracks,
    )

    players_count = 0

    for box in player_boxes:

        cls = int(box.cls)

        label = player_results[0].names[cls]

        if label == "Player":
            players_count += 1

        if label != "Player":
            continue

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

        top_tracks.setdefault(track_id, []).append(mapped)

        renderer.draw_player(
            field_frame,
            mapped,
            track_id,
        )

    for detection in ball_detections:

        renderer.draw_ball(
            video_frame,
            detection["center"],
        )

        mapped = homography.transform(
            detection["center"],
        )

        renderer.draw_ball(
            field_frame,
            mapped,
            radius=6,
            label=False,
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
        f"Jugadores: {players_count}",
        (20, 105),
    )

    renderer.draw_text(
        field_frame,
        f"Jugadores: {players_count }",
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
print("BALL DETECTOR REPORT")
print("=" * 60)

coverage = ball_frames / frame_number * 100

print(f"Frames con pelota : {ball_frames}")
print(f"Frames sin pelota : {ball_missed_frames}")
print(f"Cobertura         : {coverage:.2f}%")

if total_ball_detections:

    print(
        "Confianza media  :",
        f"{total_ball_confidence / total_ball_detections:.3f}",
    )
