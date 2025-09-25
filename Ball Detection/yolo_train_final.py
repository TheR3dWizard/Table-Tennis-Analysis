!pip install ultralytics

from ultralytics import YOLO
import os, random, shutil
from google.colab import drive

# ========================
# 1. Mount Google Drive
# ========================
drive.mount('/content/drive')

# ========================
# 2. Paths
# ========================
dataset_path = "/content/drive/MyDrive/Table Tennis/dataset"
project_path = "/content/drive/MyDrive/Table Tennis/YOLOv11_results"
subset_path = "/content/drive/MyDrive/Table Tennis/dataset_subset"

# ========================
# 3. Create subset folders
# ========================
def create_subset(src_img, src_lbl, dst_img, dst_lbl, n):
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    images = [f for f in os.listdir(src_img) if f.endswith((".jpg", ".png"))]
    random.shuffle(images)
    subset = images[:n]

    for img in subset:
        base = os.path.splitext(img)[0]
        lbl = base + ".txt"

        # copy image
        shutil.copy(os.path.join(src_img, img), os.path.join(dst_img, img))
        # copy label if exists
        if os.path.exists(os.path.join(src_lbl, lbl)):
            shutil.copy(os.path.join(src_lbl, lbl), os.path.join(dst_lbl, lbl))

# create subsets
create_subset(
    f"{dataset_path}/images/train", f"{dataset_path}/labels/train",
    f"{subset_path}/images/train", f"{subset_path}/labels/train", 2000
)
create_subset(
    f"{dataset_path}/images/val", f"{dataset_path}/labels/val",
    f"{subset_path}/images/val", f"{subset_path}/labels/val", 1000
)

print("✅ Created subset dataset at:", subset_path)

# ========================
# 4. Write data.yaml file for subset
# ========================
data_yaml = f"""
train: {subset_path}/images/train
val: {subset_path}/images/val

nc: 1
names:
  0: ball
"""

with open('/content/data.yaml', 'w') as f:
    f.write(data_yaml)

print("✅ data.yaml created at /content/data.yaml")

# ========================
# 5. Train YOLOv11 with subset
# ========================
model = YOLO('yolo11n.pt')

results = model.train(
    data='/content/data.yaml',
    epochs=10,          # now smaller dataset → increase epochs a bit
    imgsz=640,
    batch=16,
    device="cpu",           # set to "0" if GPU enabled, else "cpu"
    project=project_path,
    name='ball_detect_subset',
    save=True,
    save_period=1
)

# ========================
# 6. Validate Model
# ========================
metrics = model.val(data='/content/data.yaml')

# ========================
# 7. Run Inference on Val Set
# ========================
predictions = model.predict(
    source=f"{subset_path}/images/val",
    conf=0.25,
    save=True
)

print("✅ Training and Inference Complete!")
