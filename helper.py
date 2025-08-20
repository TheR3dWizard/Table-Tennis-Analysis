import cv2
import numpy as np

def plotpoints(image_path, points, point_color=(0, 0, 255), radius=5):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    # Plot each point
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), radius, point_color, -1)
        cv2.putText(image, f"({x},{y})", (int(x)+8, int(y)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show the image
    cv2.imshow("Image with Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
# image_path = "assets/demo/frame_0133.jpg"
# points = [(100, 150), (200, 250), (300, 350)]  # Replace with your points
# plotpoints(image_path, points)