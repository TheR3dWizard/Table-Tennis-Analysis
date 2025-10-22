import cv2
import json

# Inputs
video_path = "v1.mp4"
json_path = "ball_trajectory_v1_offset.json"
output_path = "v1_actual_ball_markup.mp4"

start_frame = 0
end_frame = 372  # inclusive
segment_name = "s1"
 

# Offset values to correct ball position
x_offset = 0  # adjust based on testing
y_offset = 0  # adjust based on testing

# Load ball markup data
with open(json_path, "r") as f:
    ball_data = json.load(f)

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and writer for output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Set the video to the start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Process frames
frame_no = start_frame
while frame_no <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    # If ball data exists for this frame
    str_frame = str(frame_no)
    if str_frame in ball_data:
        x, y = ball_data[str_frame]["x"], ball_data[str_frame]["y"]

        # Draw ball if valid coordinates
        if x >= 0 and y >= 0:
            cv2.circle(frame, (x + x_offset, y + y_offset), 8, (0, 0, 255), -1)  # red circle
            cv2.putText(frame, "ball", (x + x_offset + 10, y + y_offset - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)
    frame_no += 1

cap.release()
out.release()
print(f"Segment '{segment_name}' saved as {output_path}")