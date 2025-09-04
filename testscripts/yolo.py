'''
Image and Video Object Detection using YOLOv8
Does not provide any meaningful detection
'''

# Load YOLOv8 model (requires ultralytics package)
from ultralytics import YOLO
import cv2

def process_video_with_yolo(video_path, table_class_name='table', human_class_name='person',
                            ball_class_name='sports ball', racket_class_names=('table tennis racket', 'tennis racket')):
    # Load a pretrained YOLOv8 model (change path if you have a custom model)
    model = YOLO('yolov8n.pt')  # Use yolov8n.pt for speed, or your custom weights

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame)
        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == table_class_name:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif label == human_class_name:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif label == ball_class_name:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            elif label in racket_class_names:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        cv2.imshow('YOLO Table, Human, Ball & Racket Detection', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image_with_yolo(image, table_class_name='table', human_class_name='person',
                            ball_class_name='sports ball', racket_class_names=('table tennis racket', 'tennis racket')):
    # Load a pretrained YOLOv8 model (change path if you have a custom model)
    model = YOLO('yolov8n.pt')  # Use yolov8n.pt for speed, or your custom weights

    # Run YOLO inference
    results = model(image)
    boxes = results[0].boxes
    names = results[0].names

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == table_class_name:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif label == human_class_name:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif label == ball_class_name:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        elif label in racket_class_names:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow('YOLO Table, Human, Ball & Racket Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage:
img = cv2.imread('Heatmap/i3.png')
process_image_with_yolo(
    img,
    table_class_name='table',
    human_class_name='person',
    ball_class_name='sports ball',
    racket_class_names=('table tennis racket', 'tennis racket')
)