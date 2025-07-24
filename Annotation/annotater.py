import cv2
import csv
import time

video_path = 'game_1.mp4'  
output_csv = 'annotations.csv'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

annotations = []
start_time = None

def frame_to_time(frame_num):
    seconds = frame_num / fps
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

print("[INFO] Controls:")
print("  Press 's' to mark start of event")
print("  Press 'e' to mark end of event")
print("  Press 'q' to quit and save")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    time_str = frame_to_time(current_frame)
    cv2.putText(frame, f"Time: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video Annotator', frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        start_time = current_frame
        print(f"Start marked at {start_time}")

    elif key == ord('e') and start_time:
        end_time = current_frame
        label = input(f"Enter label for event from {start_time} to {end_time}: ")
        annotations.append([video_path, start_time, end_time, label])
        print(f"Event saved: {label} from {start_time} to {end_time}")
        start_time = None

    elif key == ord('q'):
        print("Saving annotations...")
        break

cap.release()
cv2.destroyAllWindows()

# Save annotations
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "start_time", "end_time", "event_label"])
    writer.writerows(annotations)

print(f"Annotations saved to {output_csv}")
