import cv2
import csv
import time

#config
video_path = "game_1.mp4"
output_csv = "interpolated_annotations.csv"
instructions = "SPACE=pause | a/d=back/fwd | q=quit | Click: P1_start, P2_start, P1_end, P2_end"

#initiate
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Couldn't open video")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
paused = False
annotations = []
click_buffer = []  # (frame_idx, x, y, label)
interpolation_set = []  # Temporarily holds the 4 points for interpolation

def frame_to_time(frame_idx):
    seconds = frame_idx / fps
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def draw_overlay(frame, frame_idx):
    time_str = frame_to_time(frame_idx)
    cv2.putText(frame, f"Time: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, instructions, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

def click_callback(event, x, y, flags, param):
    global interpolation_set
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if len(interpolation_set) == 0:
            label = "Player1_start"
        elif len(interpolation_set) == 1:
            label = "Player2_start"
        elif len(interpolation_set) == 2:
            label = "Player1_end"
        elif len(interpolation_set) == 3:
            label = "Player2_end"
        else:
            return

        interpolation_set.append((frame_idx, x, y, label))
        print(f"[{label}] at frame {frame_idx} → ({x},{y})")

cv2.namedWindow("Video Annotator")
cv2.setMouseCallback("Video Annotator", click_callback)

def interpolate_points(p_start, p_end, label):
    annotations_local = []
    f1, x1, y1, _ = p_start
    f2, x2, y2, _ = p_end
    for f in range(f1, f2 + 1):
        alpha = (f - f1) / (f2 - f1) if f2 != f1 else 0
        x = int(x1 + (x2 - x1) * alpha)
        y = int(y1 + (y2 - y1) * alpha)
        t_str = frame_to_time(f)
        annotations_local.append([video_path, f, t_str, x, y, label])
    return annotations_local

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break
    else:
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    draw_overlay(frame, frame_idx)
    frame = cv2.resize(frame, (800, 600))
    display_frame = frame.copy()

    # Draw points from click buffer
    for _, x, y, label in interpolation_set:
        color = (0, 255, 0) if "Player1" in label else (0, 0, 255)
        cv2.circle(display_frame, (x, y), 5, color, -1)
        cv2.putText(display_frame, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Video Annotator", display_frame)
    key = cv2.waitKey(int(1000 / fps)) & 0xFF

    if key == ord(' '):  # Pause/Resume
        paused = not paused

    elif key == ord('a'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - int(fps * 2)))

    elif key == ord('d'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, frame_idx + int(fps * 2)))

    elif key in [ord('q'), 27]:
        break

    # Once all 4 clicks are done → interpolate and save
    if len(interpolation_set) == 4:
        print("[Interpolating between frames]")
        p1_start = interpolation_set[0]
        p2_start = interpolation_set[1]
        p1_end = interpolation_set[2]
        p2_end = interpolation_set[3]

        annotations += interpolate_points(p1_start, p1_end, "Player1")
        annotations += interpolate_points(p2_start, p2_end, "Player2")

        interpolation_set = []

cap.release()
cv2.destroyAllWindows()

#save csv
with open(output_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(annotations)

print(f"✅ Saved {len(annotations)} interpolated annotations to {output_csv}")
