from pathlib import Path
import cv2
from ultralytics import YOLO
import csv
import time

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "partido_detectado.mp4"

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(width, height, fps)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

MODEL_PATH = BASE_DIR / "models" / "yolo11l.pt"

model = YOLO(str(MODEL_PATH))

frame_number = 0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

track_stats = {}
tracks = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    print(f"\rProcesando frame {frame_number}/{total_frames}", end="")

    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.25,
        imgsz=960,
        verbose=False,
    )

    annotated_frame = frame.copy()

    boxes = results[0].boxes

    for box in boxes:

        if box.id is None:
            continue

        track_id = int(box.id)

        confidence = float(box.conf)

        if track_id not in track_stats:

            track_stats[track_id] = {
                "first_frame": frame_number,
                "last_frame": frame_number,
                "frames_seen": 1,
                "conf_sum": confidence,
                "last_seen": frame_number,
                "missing_frames": 0,
            }

        else:

            track_stats[track_id]["last_frame"] = frame_number
            track_stats[track_id]["frames_seen"] += 1
            track_stats[track_id]["conf_sum"] += confidence
            gap = frame_number - track_stats[track_id]["last_seen"] - 1

            track_stats[track_id]["missing_frames"] += max(gap, 0)

            track_stats[track_id]["last_seen"] = frame_number

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            annotated_frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    for box in boxes:

        if box.id is None:
            continue

        track_id = int(box.id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        tracks.setdefault(track_id, []).append(
            {"frame": frame_number, "center": (cx, cy)}
        )

    cv2.imshow("Football Analysis Lab", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    writer.write(annotated_frame)

csv_path = BASE_DIR / "outputs" / "tracking_report.csv"

with open(csv_path, "w", newline="") as f:

    writer_csv = csv.writer(f)

    writer_csv.writerow(
        [
            "track_id",
            "first_frame",
            "last_frame",
            "frames_seen",
            "duration_seconds",
            "average_confidence",
            "missing_frames",
        ]
    )

    for track_id, stats in track_stats.items():

        duration = stats["frames_seen"] / fps
        avg_conf = stats["conf_sum"] / stats["frames_seen"]

        writer_csv.writerow(
            [
                track_id,
                stats["first_frame"],
                stats["last_frame"],
                stats["frames_seen"],
                round(duration, 2),
                round(avg_conf, 3),
            ]
        )

cap.release()
print("\n" + "=" * 50)
print("TRACKING REPORT")
print("=" * 50)

print(f"Frames procesados: {frame_number}")
print(f"IDs creados: {len(track_stats)}")
long_tracks = 0
short_tracks = 0

max_track = None
max_frames = 0

for track_id, stats in track_stats.items():

    if stats["frames_seen"] >= 100:
        long_tracks += 1

    if stats["frames_seen"] <= 10:
        short_tracks += 1

    if stats["frames_seen"] > max_frames:
        max_frames = stats["frames_seen"]
        max_track = track_id

print(f"Tracks largos: {long_tracks}")
print(f"Tracks cortos: {short_tracks}")
start = time.perf_counter()

end = time.perf_counter()

elapsed = end - start

print(f"Tiempo total: {elapsed:.2f} s")
print(f"Tiempo por frame: {elapsed / frame_number:.4f} s")
print(f"FPS efectivos: {frame_number / elapsed:.2f}")
print(f"Track más largo: ID {max_track} ({max_frames} frames)")
writer.release()
cv2.destroyAllWindows()
print("Video generado correctamente.")
