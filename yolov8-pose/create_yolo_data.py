import csv
from collections import defaultdict

# Load your CSV-like data
with open("point_labels.csv", "r") as f:
    reader = csv.reader(f)
    data = list(reader)

# Group points by image
images = defaultdict(list)
for row in data:
    label, x, y, img, W, H = row
    x, y, W, H = map(int, [x, y, W, H])
    images[img].append((label, x, y, W, H))

# Process each image
for img, points in images.items():
    W, H = points[0][3], points[0][4]
    coords = [(x, y) for (_, x, y, _, _) in points]

    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2 / W
    y_center = (y_min + y_max) / 2 / H
    bbox_w = (x_max - x_min) / W
    bbox_h = (y_max - y_min) / H

    norm_points = [(x / W, y / H) for x, y in coords]

    # YOLO format line
    line = f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} " + \
           " ".join([f"{px:.6f} {py:.6f}" for (px, py) in norm_points])

    txt_file = img.replace(".jpg", ".txt")
    with open(txt_file, "w") as f:
        f.write(line + "\n")

    print(f"Saved {txt_file}")
