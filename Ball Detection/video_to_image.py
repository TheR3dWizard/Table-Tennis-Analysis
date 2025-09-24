import cv2
import os

video_path = "game_1.mp4"
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_id += 1

cap.release()
print(f"Extracted {frame_id} frames to {output_dir}")
