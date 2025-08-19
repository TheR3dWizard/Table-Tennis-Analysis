import cv2
import numpy as np

# Load video
video_path = 'game_1.mp4'
cap = cv2.VideoCapture(video_path)

# Example: List of values for each frame
# (replace this with your actual list)
values = np.load("outputs_game_1_0.npy",allow_pickle=True).tolist()
values1 = np.load("outputs_game_1_1.npy",allow_pickle=True).tolist()
values.extend(values1)
# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Green
thickness = 2
position = (10, 30)  # Top-left corner

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(values):
        break

    # Overlay the value on the frame
    cv2.putText(frame, str(values[frame_idx]), position, font, font_scale, color, thickness)

    # Display the frame
    cv2.imshow('Video with Values', frame)

    # Wait 25 ms between frames or break on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
