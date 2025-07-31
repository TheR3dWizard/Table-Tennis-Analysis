import cv2
import numpy as np
import time

from rtmlib import Wholebody, draw_skeleton

device = 'cpu'
backend = 'onnxruntime'
video_path = './demo.mp4'  # Path to your video file

openpose_skeleton = False

wholebody = Wholebody(
    to_openpose=openpose_skeleton,
    mode='lightweight',  # 'performance' or 'lightweight' for faster inference
    backend=backend,
    device=device
)

cap = cv2.VideoCapture(video_path)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    keypoints, scores = wholebody(frame)
    img_show = frame.copy()
    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.5)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-8)
    cv2.putText(img_show, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()