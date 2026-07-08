from pathlib import Path
import cv2

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

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

from ultralytics import YOLO

MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"

model = YOLO(str(MODEL_PATH))

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"\rProcesando frame {frame_number}/{total_frames}",
        end=""
    )

    results = model.predict(
        source=frame,
        classes=[0],
        conf=0.25,
        imgsz=960,
        verbose=False
    )

    annotated_frame = results[0].plot()

    boxes = results[0].boxes

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Players: {len(boxes)}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("Football Analysis Lab", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    writer.write(annotated_frame)

cap.release()
writer.release()
cv2.destroyAllWindows()
print("Video generado correctamente.")