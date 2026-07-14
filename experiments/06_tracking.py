from pathlib import Path
import csv
import time

import cv2

from src.video.video_processor import VideoProcessor
from src.detection.player_detector import YOLODetector

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "partido_detectado.mp4"
MODEL_PATH = BASE_DIR / "models" / "yolo11m.pt"


processor = VideoProcessor(
    VIDEO_PATH,
    OUTPUT_PATH,
)

detector = YOLODetector(
    MODEL_PATH,
    tracker="bytetrack.yaml",
)

print(
    processor.width,
    processor.height,
    processor.fps,
)

frame_number = 0

track_stats = {}
tracks = {}

start = time.perf_counter()

while True:

    ret, frame = processor.read()

    if not ret:
        break

    frame_number += 1

    print(f"\rProcesando frame {frame_number}/{processor.total_frames}", end="")

    results = detector.track(frame)

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

            stats = track_stats[track_id]

            stats["last_frame"] = frame_number
            stats["frames_seen"] += 1
            stats["conf_sum"] += confidence

            gap = frame_number - stats["last_seen"] - 1

            stats["missing_frames"] += max(gap, 0)
            stats["last_seen"] = frame_number

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        tracks.setdefault(track_id, []).append(
            {
                "frame": frame_number,
                "center": (cx, cy),
            }
        )

    cv2.imshow(
        "Football Analysis Lab",
        annotated_frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    processor.write(annotated_frame)

processor.release()

elapsed = time.perf_counter() - start

csv_path = BASE_DIR / "outputs" / "tracking_report.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(
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

        writer.writerow(
            [
                track_id,
                stats["first_frame"],
                stats["last_frame"],
                stats["frames_seen"],
                round(stats["frames_seen"] / processor.fps, 2),
                round(stats["conf_sum"] / stats["frames_seen"], 3),
                stats["missing_frames"],
            ]
        )

print("\n" + "=" * 50)
print("TRACKING REPORT")
print("=" * 50)

print(f"Frames procesados: {frame_number}")
print(f"IDs creados: {len(track_stats)}")

long_tracks = sum(1 for stats in track_stats.values() if stats["frames_seen"] >= 100)

short_tracks = sum(1 for stats in track_stats.values() if stats["frames_seen"] <= 10)

max_track = max(
    track_stats,
    key=lambda k: track_stats[k]["frames_seen"],
)

print(f"Tracker: {detector.tracker}")

print(f"Tracks largos: {long_tracks}")
print(f"Tracks cortos: {short_tracks}")

print(f"Tiempo total: {elapsed:.2f} s")
print(f"Tiempo por frame: {elapsed/frame_number:.4f} s")
print(f"FPS efectivos: {frame_number/elapsed:.2f}")

print(
    f"Track más largo: ID {max_track} "
    f"({track_stats[max_track]['frames_seen']} frames)"
)

print("Video generado correctamente.")
