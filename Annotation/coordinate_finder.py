import cv2

# Mouse callback function
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at: x={x}, y={y}")
        # Optionally, draw the point on the image
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", param)

# Load an image
def get_frame_from_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return None
    return frame

video_path = "../Videos/game_1.mp4"  # Replace with your video path
frame_number = 100  # Replace with your desired frame number
img = get_frame_from_video(video_path, frame_number)

if img is None:
    print("Error: Could not read image.")
    exit()

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", show_coordinates, img)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
