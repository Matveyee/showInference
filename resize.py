import cv2
import sys
import os

# Входной аргумент — путь к видео
if len(sys.argv) != 2:
    print("Usage: python resize_video.py input_video.mp4")
    sys.exit(1)

input_path = sys.argv[1]
output_path = os.path.splitext(input_path)[0] + "_resized_640x640.mp4"

# Целевой размер
WIDTH, HEIGHT = 640, 640

# Открываем входное видео
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: cannot open video")
    sys.exit(1)

# Получаем FPS и fourcc
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Создаём выходной видеофайл
out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    out.write(resized)
    frame_count += 1

cap.release()
out.release()

print(f"Готово: сохранено {frame_count} кадров в {output_path}")