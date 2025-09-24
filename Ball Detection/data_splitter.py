import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths
images_path = "frames"
labels_path = "labels"
output_path = "dataset"

# Train/Val split ratio
val_ratio = 0.2

# Get all images (assuming .jpg or .png)
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]

# Split train and val
train_files, val_files = train_test_split(image_files, test_size=val_ratio, random_state=42)

# Function to copy files
def copy_files(file_list, split):
    for f in file_list:
        # image
        src_img = os.path.join(images_path, f)
        dst_img = os.path.join(output_path, "images", split, f)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy(src_img, dst_img)

        # label (same name, .txt)
        label_file = os.path.splitext(f)[0] + ".txt"
        src_lbl = os.path.join(labels_path, label_file)
        if os.path.exists(src_lbl):
            dst_lbl = os.path.join(output_path, "labels", split, label_file)
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            shutil.copy(src_lbl, dst_lbl)

# Copy train and val
copy_files(train_files, "train")
copy_files(val_files, "val")

print(f"✅ Dataset prepared in {output_path}")
