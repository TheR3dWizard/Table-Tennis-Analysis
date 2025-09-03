import cv2
import numpy as np
import time

def player_finding_and_midpoint(background_frame, current_frame, PSI):
    background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    background_blur = cv2.GaussianBlur(background_gray, (5, 5), 0)
    current_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
    diff_image = cv2.absdiff(background_blur, current_blur)
    mean_val = np.mean(diff_image)
    std_val = np.std(diff_image)
    threshold_value = mean_val + std_val / PSI
    _, binary_mask = cv2.threshold(diff_image, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            player_contour = cnt
    if player_contour is not None:
        x, y, w, h = cv2.boundingRect(player_contour)
        midpoint = (int(x + w / 2), int(y + h / 2))
        return (x, y, w, h), midpoint, closed_mask
    else:
        return None, None, closed_mask

def process_video(video_path, background_path, PSI=4):
    cap = cv2.VideoCapture(video_path)
    background_frame = cv2.imread(background_path)
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        bbox, midpoint, mask = player_finding_and_midpoint(background_frame, frame, PSI)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, midpoint, 5, (0, 0, 255), -1)
        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time + 1e-8)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red
        print(f"FPS: {fps:.2f}")
        cv2.imshow('Player Detection', frame)
        cv2.imshow('Player Mask', mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
process_video('assets/demo.mp4', 'assets/bg4.png', PSI=4)