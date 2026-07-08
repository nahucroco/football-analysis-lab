from pathlib import Path
import json

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "outputs" / "frame.png"

CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)

CONFIG_PATH = CONFIG_DIR / "camera_calibration.json"


# --------------------------------------------------
# Cargar imagen
# --------------------------------------------------

image = cv2.imdecode(
    np.fromfile(IMAGE_PATH, dtype=np.uint8),
    cv2.IMREAD_COLOR,
)

if image is None:
    raise RuntimeError(f"No se pudo abrir {IMAGE_PATH}")

height, width = image.shape[:2]

points = []


# --------------------------------------------------
# Dibujar interfaz
# --------------------------------------------------

def redraw():

    canvas = image.copy()

    instructions = [
        "LAB 09 - CAMERA CALIBRATION",
        "",
        "Click izquierdo -> Agregar punto",
        "ENTER           -> Guardar",
        "BACKSPACE       -> Eliminar ultimo",
        "R               -> Reiniciar",
        "ESC             -> Salir",
        "",
        f"Puntos seleccionados: {len(points)}",
        "Minimo requerido: 4",
    ]

    y = 30

    for text in instructions:

        cv2.putText(
            canvas,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        y += 25

    # Dibujar puntos

    for i, (x, y) in enumerate(points):

        cv2.circle(
            canvas,
            (x, y),
            5,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            canvas,
            str(i + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # Dibujar polígono

    if len(points) >= 2:

        for i in range(len(points) - 1):

            cv2.line(
                canvas,
                points[i],
                points[i + 1],
                (0, 255, 0),
                2,
            )

    return canvas


display = redraw()


# --------------------------------------------------
# Mouse
# --------------------------------------------------

def mouse_callback(event, x, y, flags, param):

    global display

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((x, y))

    print(f"Punto {len(points)} -> ({x}, {y})")

    display = redraw()


cv2.namedWindow("Calibration")

cv2.setMouseCallback(
    "Calibration",
    mouse_callback,
)


# --------------------------------------------------
# Loop principal
# --------------------------------------------------

while True:

    cv2.imshow(
        "Calibration",
        display,
    )

    key = cv2.waitKey(20) & 0xFF

    # ESC

    if key == 27:
        break

    # ENTER

    elif key == 13:

        if len(points) < 4:

            print("\nSe necesitan al menos 4 puntos.\n")
            continue

        data = {

            "image_width": width,
            "image_height": height,

            "image_points": points

        }

        with open(
            CONFIG_PATH,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
            )

        print()
        print("=" * 60)
        print("CALIBRACION GUARDADA")
        print("=" * 60)
        print(CONFIG_PATH)

        break

    # BACKSPACE

    elif key == 8:

        if points:

            removed = points.pop()

            print(f"Punto eliminado {removed}")

            display = redraw()

    # R

    elif key == ord("r"):

        points.clear()

        print("Puntos reiniciados.")

        display = redraw()

cv2.destroyAllWindows()