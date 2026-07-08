from pathlib import Path
import json

import cv2
import numpy as np

from src.calibration.field import FootballField


BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "outputs" / "frame.png"

CONFIG_DIR = BASE_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)

CONFIG_PATH = CONFIG_DIR / "camera_calibration.json"

FIELD_CONFIG_PATH = CONFIG_DIR / "field_5v5.json"

FIELD_IMAGE_PATH = (
    BASE_DIR /
    "assets" /
    "fields" /
    "football5.png"
)

SEPARATOR = 20


# ==========================================================
# Cargar imagen del video
# ==========================================================

video = cv2.imdecode(
    np.fromfile(IMAGE_PATH, dtype=np.uint8),
    cv2.IMREAD_COLOR,
)

if video is None:
    raise RuntimeError(f"No se pudo abrir {IMAGE_PATH}")


video_height, video_width = video.shape[:2]


# ==========================================================
# Crear cancha
# ==========================================================

field = FootballField(
    FIELD_CONFIG_PATH,
    render_height_px=video_height,
)

field_image = field.load_image(
    FIELD_IMAGE_PATH,
)

field_height, field_width = field_image.shape[:2]


# ==========================================================
# Estado
# ==========================================================

image_points = []
field_points = []

current_image_point = None

waiting_video = True


# ==========================================================
# Dibujar interfaz
# ==========================================================

def draw_points(image, points):

    for i, (x, y) in enumerate(points):

        cv2.circle(
            image,
            (x, y),
            5,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            image,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    if len(points) >= 2:

        for i in range(len(points) - 1):

            cv2.line(
                image,
                points[i],
                points[i + 1],
                (0, 255, 0),
                2,
            )

    if len(points) >= 3:

        cv2.line(
            image,
            points[-1],
            points[0],
            (0, 255, 0),
            2,
        )

def redraw():

    video_canvas = video.copy()
    field_canvas = field_image.copy()

    # Dibujar puntos del video

    draw_points(
        video_canvas,
        image_points,
    )

    if current_image_point is not None:

        cv2.circle(
            video_canvas,
            current_image_point,
            8,
            (255, 0, 0),
            2,
        )

    draw_points(
        field_canvas,
        field_points,
    )
    # Unir imágenes

    separator = np.full(
        (
            video_height,
            SEPARATOR,
            3,
        ),
        40,
        dtype=np.uint8,
    )

    canvas = np.hstack(
        (
            video_canvas,
            separator,
            field_canvas,
        )
    )

    # Panel de ayuda

    instructions = [

        "LAB 10 - CALIBRATION TOOL",

        "",

        f"Pares completados : {len(field_points)}",
        f"Puntos de video   : {len(image_points)}",
        f"Puntos de cancha  : {len(field_points)}",

        "",

        "Flujo:",
        "1. Click en VIDEO",
        "2. Click en CANCHA",

        "",

        "ENTER     Guardar",
        "BACKSPACE Eliminar ultimo",
        "R         Reiniciar",
        "ESC       Salir",

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

    next_point = max(
        len(image_points),
        len(field_points),
    ) + 1

    status = (
        f"Seleccione el punto #{next_point} en el VIDEO"
        if waiting_video
        else f"Seleccione el punto #{next_point} en la CANCHA"
    )

    status_color = (
        (0, 255, 255)
        if waiting_video
        else (255, 255, 0)
    )

    cv2.putText(
        canvas,
        status,
        (20, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    return canvas


display = redraw()


# ==========================================================
# Mouse
# ==========================================================

def mouse_callback(event, x, y, flags, param):

    global waiting_video
    global display
    global current_image_point

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # ============================================
    # Esperando un punto del VIDEO
    # ============================================

    if waiting_video:

        if x >= video_width:

            print("Primero seleccione un punto del VIDEO.")

            return

        current_image_point = (x, y)

        waiting_video = False

        print(
            f"VIDEO -> ({x}, {y})"
        )

    # ============================================
    # Esperando un punto de la CANCHA
    # ============================================

    else:

        if x < video_width + SEPARATOR:
            return

        fx = x - video_width - SEPARATOR

        if fx >= field_width:
            return

        if y >= field_height:
            return

        image_points.append(current_image_point)

        field_points.append((fx, y))

        print(
            f"FIELD -> ({fx}, {y})"
        )

        current_image_point = None

        waiting_video = True

    display = redraw()

# ==========================================================
# Ventana
# ==========================================================

cv2.namedWindow("Calibration Tool")

cv2.setMouseCallback(
    "Calibration Tool",
    mouse_callback,
)


# ==========================================================
# Loop principal
# ==========================================================

while True:

    cv2.imshow(
        "Calibration Tool",
        display,
    )

    key = cv2.waitKey(20) & 0xFF

    # ESC

    if key == 27:
        break

    # ENTER

    elif key == 13:

        if len(image_points) < 4:

            print("Se necesitan al menos 4 pares.")

            continue

        if len(image_points) != len(field_points):

            print("Falta completar un par.")

            continue

        data = {

            "field_type": "field_5v5",

            "image_width": video_width,
            "image_height": video_height,

            "field_width": field_width,
            "field_height": field_height,

            "field_width_m": field.field_width_m,
            "field_height_m": field.field_height_m,

            "image_points": image_points,
            "field_points": field_points,

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

        print("=" * 50)
        print("CALIBRACION GUARDADA")
        print("=" * 50)
        print(CONFIG_PATH)

        break

    # BACKSPACE

    elif key == 8:

        if not waiting_video:

            current_image_point = None
            waiting_video = True

            print("Último punto del VIDEO eliminado.")

        elif image_points:

            image_points.pop()
            field_points.pop()

            print("Último par eliminado.")

        display = redraw()

    # R

    elif key == ord("r"):

        image_points.clear()
        field_points.clear()

        waiting_video = True

        display = redraw()

        print("Calibración reiniciada.")

cv2.destroyAllWindows()