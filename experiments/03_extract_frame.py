from pathlib import Path
import cv2

# ==========================================
# Configuración de rutas
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "partido.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "frame.png"

# Crear la carpeta de salida si no existe
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Video: {VIDEO_PATH}")
print(f"Salida: {OUTPUT_PATH}")

# ==========================================
# Abrir video
# ==========================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

# ==========================================
# Leer el primer frame
# ==========================================

ret, frame = cap.read()

if not ret:
    print("❌ No se pudo leer el primer frame.")
    cap.release()
    exit()

print("Frame leído correctamente")
print("Shape:", frame.shape)
print("Tipo:", frame.dtype)

# ==========================================
# Guardar imagen
# ==========================================

from PIL import Image

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

Image.fromarray(frame_rgb).save(OUTPUT_PATH)

print("✅ Frame guardado con Pillow.")