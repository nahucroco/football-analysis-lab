from pathlib import Path
import cv2

# ==========================
# Configuración
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"

# ==========================
# Abrir video
# ==========================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

print("Video abierto correctamente.")

# ==========================
# Información del video
# ==========================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Resolución: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Cantidad de frames: {frame_count}")
print(f"Duración: {duration:.2f} segundos")

# ==========================
# Procesamiento frame por frame
# ==========================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    # Punto rojo
    cv2.circle(
        frame,
        (center_x, center_y),
        8,
        (0, 0, 255),
        -1
    )

    # Cruz azul
    cv2.line(
        frame,
        (center_x - 20, center_y),
        (center_x + 20, center_y),
        (255, 0, 0),
        2
    )

    cv2.line(
        frame,
        (center_x, center_y - 20),
        (center_x, center_y + 20),
        (255, 0, 0),
        2
    )

    # Rectángulo verde
    cv2.rectangle(
        frame,
        (100, 100),
        (250, 300),
        (0, 255, 0),
        3
    )

    # Número de frame
    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Football Analysis Lab", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()