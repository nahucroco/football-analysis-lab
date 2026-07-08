from pathlib import Path

import cv2
from ultralytics import YOLO

# ==========================
# Configuración
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"
MODEL_PATH = BASE_DIR / "models" / "yolo11s.pt"

# ==========================
# Abrir video
# ==========================

cap = cv2.VideoCapture(str(VIDEO_PATH))

ret, frame = cap.read()

cap.release()

if not ret:
    raise RuntimeError("No se pudo leer el primer frame.")

# ==========================
# Cargar modelo
# ==========================

model = YOLO(str(MODEL_PATH))

# ==========================
# Inferencia
# ==========================

""" results = model.predict(
    source=frame,
    classes=[0,32],
    conf=0.25,
    iou=0.45,
    imgsz=960,
    save=True
) """

import time

start = time.perf_counter()

results = model.predict(
    source=frame,
    imgsz=960,
    conf=0.15,
    iou=0.45,
    classes=[0],
    save=True
)

elapsed = time.perf_counter() - start

boxes = results[0].boxes

print("=" * 40)
print("RESULTADOS DEL EXPERIMENTO")
print("=" * 40)

print(f"Modelo: yolo11s")
print(f"Tiempo total: {elapsed:.3f}s")
print(f"Objetos detectados: {len(boxes)}")

confianzas = [float(box.conf) for box in boxes]

print(f"Confianza media: {sum(confianzas)/len(confianzas):.2f}")

# ==========================
# Mostrar resultados
# ==========================

for result in results:

    print(f"\nSe detectaron {len(result.boxes)} objetos\n")

    for box in result.boxes:

        cls = int(box.cls)

        print(
            f"{model.names[cls]} | "
            f"Confianza: {float(box.conf):.2f}"
        )