import os
import pandas as pd
from PIL import Image

# Paths
frames_dir = "frames"   # folder with frame_0001.jpg ...
labels_dir = "labels"   # output YOLO labels
os.makedirs(labels_dir, exist_ok=True)

# Load ball annotations
# CSV format: frame,x,y,x_min,y_min,x_max,y_max
df = pd.read_csv("ball_bounding_boxes_game1.csv")

# Group annotations by frame number
annotations = {}
for _, row in df.iterrows():
    fid = int(row["frame"])
    if fid not in annotations:
        annotations[fid] = []
    annotations[fid].append(row)

for frame_file in sorted(os.listdir(frames_dir)):
    if not frame_file.endswith(".jpg"):
        continue

    # Extract frame number from file name (frame_0001.jpg → 1)
    frame_id = int(frame_file.split("_")[-1].split(".")[0])

    frame_path = os.path.join(frames_dir, frame_file)
    img = Image.open(frame_path)
    w, h = img.size

    label_path = os.path.join(labels_dir, frame_file.replace(".jpg", ".txt"))

    if frame_id in annotations:
        with open(label_path, "w") as f:
            for row in annotations[frame_id]:
                x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

                # Convert to YOLO format
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                box_w = (x_max - x_min) / w
                box_h = (y_max - y_min) / h

                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
    else:
        # Frame has no annotation → empty label file
        open(label_path, "w").close()
