import cv2
import pandas as pd
import os

def draw_bounding_boxes(video_path, csv_path, output_path="output_video.mp4"):
    # Read bounding box data
    # CSV should have: frame,x_min,y_min,x_max,y_max
    df = pd.read_csv(csv_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Video writer setup (same resolution + FPS as input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Get all bounding boxes for this frame
        boxes = df[df['frame'] == frame_num]

        for _, row in boxes.iterrows():
            x_min, y_min, x_max, y_max = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
            # Draw bounding box (green)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Label
            cv2.putText(frame, f"Ball", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Write frame with bounding box
        out.write(frame)

        # Optional: show in window
        cv2.imshow("Bounding Box Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_path}")



video_path = "game_1.mp4"  
csv_path = "ball_bounding_boxes.csv"  
output_path = "video_with_boxes.mp4"

draw_bounding_boxes(video_path, csv_path, output_path)
