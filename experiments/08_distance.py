from pathlib import Path
import time
import csv
import cv2

from src.video.video_processor import VideoProcessor
from src.detection.yolo_detector import YOLODetector
from src.visualization.renderer import Renderer
from src.analytics.metrics import Metrics


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "partido_distance.mp4"
MODEL_PATH = BASE_DIR / "models" / "yolo11m.pt"


processor = VideoProcessor(
    VIDEO_PATH,
    OUTPUT_PATH,
)

detector = YOLODetector(
    MODEL_PATH,
    tracker="bytetrack.yaml",
)

renderer = Renderer()
metrics = Metrics()

tracks = {}

frame_number = 0

start = time.perf_counter()

while True:

    ret, frame = processor.read()

    if not ret:
        break

    frame_number += 1

    print(
        f"\rProcesando frame {frame_number}/{processor.total_frames}",
        end=""
    )

    results = detector.track(frame)

    boxes = results[0].boxes

    #
    # Actualizar trayectorias
    #

    for box in boxes:

        if box.id is None:
            continue

        track_id = int(box.id)

        confidence = float(box.conf)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        tracks.setdefault(track_id, []).append(
            {
                "frame": frame_number,
                "center": (cx, cy),
                "confidence": confidence,
            }
        )

    annotated_frame = frame.copy()

    #
    # Dibujar trayectorias
    #

    annotated_frame = renderer.draw_trajectories(
        annotated_frame,
        tracks,
        history=40,
    )

    #
    # Dibujar cajas
    #

    annotated_frame = renderer.draw_boxes(
        annotated_frame,
        boxes,
    )

    #
    # Información
    #

    cv2.imshow(
        "Football Analysis Lab",
        annotated_frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    processor.write(
        annotated_frame,
    )

processor.release()

elapsed = time.perf_counter() - start

distances = metrics.calculate_distance(
    tracks
)

csv_path = BASE_DIR / "outputs" / "distance_report.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "track_id",
            "distance_pixels",
        ]
    )

    for track_id, distance in sorted(
        distances.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        writer.writerow(
            [
                track_id,
                round(distance, 2),
            ]
        )

print()
print("=" * 50)
print("DISTANCE REPORT")
print("=" * 50)

for track_id, distance in sorted(
    distances.items(),
    key=lambda x: x[1],
    reverse=True,
):

    print(
        f"Jugador {track_id:>3}: "
        f"{distance:.2f} px"
    )

print()

print(f"Modelo: {detector.model_name}")
print(f"Tracker: {detector.tracker}")

print(f"Tiempo total: {elapsed:.2f} s")
print(f"FPS efectivos: {frame_number / elapsed:.2f}")

print("Reporte guardado en outputs/distance_report.csv")