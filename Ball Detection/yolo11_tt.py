from ultralytics import YOLO

# load YOLOv11 small model
model = YOLO("yolo11s.pt")  # or yolo11n.pt (nano), yolo11m.pt (medium), etc.

# Train
model.train(
    data="data.yaml",
    epochs=50,         # increase for better accuracy
    imgsz=640,         # input image size
    batch=16,
    workers=4,
    device=0           # use GPU if available
)
