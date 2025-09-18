from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("runs/pose/train5/weights/best.pt")  # load an official model

cap = cv2.VideoCapture("../Videos/rallies_01.mp4")   # Replace with "video.mp4" for a file



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, stream=True, device="cuda")  # stream=True gives generator for efficiency

    results = model(frame, device="cuda")

    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLO Pose - Full", annotated_frame)


    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()