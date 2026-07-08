# Laboratorio 01.1 - Procesamiento básico de video con OpenCV

## Objetivo

Comprender cómo OpenCV representa un video y cómo modificar cada uno de sus frames.

## Tecnologías

- Python
- OpenCV

## Resultados

- Video abierto correctamente.
- Lectura frame por frame.
- Obtención de propiedades del video.
- Dibujo de texto, líneas, círculos y rectángulos.

## Hipótesis
- Hipótesis 1 El detector pierde jugadores: Si en algunos frames un jugador desaparece de las detecciones, ByteTrack crea un ID nuevo cuando reaparece.
- Hipótesis 2 ByteTrack necesita ajustes: Todavía estamos usando el archivo bytetrack.yaml por defecto. Tiene parámetros que podrían ajustarse para fútbol.
- Hipótesis 3 ByteTrack no es el mejor tracker para este escenario: Ultralytics también soporta BoT-SORT. Es otro tracker, diseñado para ser más robusto en algunas situaciones porque incorpora información de apariencia además de movimiento.
- Hipótesis 4 Necesitamos lógica propia

## Conclusiones

OpenCV representa un video como una secuencia de imágenes independientes. Cada frame puede procesarse y modificarse antes de mostrarse o almacenarse, constituyendo la base sobre la que posteriormente se aplicarán modelos de visión por computadora como YOLO para la detección de jugadores y otros elementos del partido.