import cv2
import csv
import time

# === CONFIG ===
video_path = "game_1.mp4"
output_csv = "annotations.csv"
skip_seconds = 2
instructions = "SPACE=pause/resume | s=start | e=end | a/←=back | d/→=fwd | q=quit"

# === INIT ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Couldn't open video")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
playback_speed = 1  # normal
paused = False
start_time = None
annotations = []

def frame_to_time(frame_idx):
    seconds = frame_idx / fps
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def draw_overlay(frame, frame_idx):
    time_str = frame_to_time(frame_idx)
    cv2.putText(frame, f"Time: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, instructions, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

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
    frame = cv2.resize(frame, (800, 600))  # Resize for better visibility
    cv2.imshow("Video Annotator", frame)

    key = cv2.waitKey(int(1000 / (fps * playback_speed))) & 0xFF

    # === CONTROLS ===
    if key == ord(' '):  # Pause/Resume
        paused = not paused

    elif key == ord('s'):
        start_time = frame_idx
        time_str = frame_to_time(start_time)
        print(f"[Start] at {time_str}")

    elif key == ord('e') and start_time:
        end_time = frame_idx
        end_time_str = frame_to_time(end_time)
        time_str = frame_to_time(start_time)
        label = input(f"Label for event {time_str} to {end_time_str}: ")
        annotations.append([video_path, start_time, end_time, label])
        print(f"[Saved] {label}: {start_time} → {end_time}")
        start_time = None

    elif key in [ord('a'), 81]:  # Left arrow / a
        back_frame = max(0, frame_idx - int(skip_seconds * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, back_frame)

    elif key in [ord('d'), 83]:  # Right arrow / d
        fwd_frame = min(total_frames - 1, frame_idx + int(skip_seconds * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fwd_frame)

    elif key == ord('+'):
        playback_speed = min(4.0, playback_speed + 0.25)
        print(f"Playback speed: {playback_speed:.2f}x")

    elif key == ord('-'):
        playback_speed = max(0.25, playback_speed - 0.25)
        print(f"Playback speed: {playback_speed:.2f}x")

    elif key in [ord('q'), 27]:  # Quit on q or ESC
        break

cap.release()
cv2.destroyAllWindows()

# === SAVE CSV ===
with open(output_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(["video_name", "start_time", "end_time", "event_label"])
    writer.writerows(annotations)

print(f"✅ Saved {len(annotations)} annotations to {output_csv}")
