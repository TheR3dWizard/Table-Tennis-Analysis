import cv2
import numpy as np
from helper import plotpoints
from constants import Constants
# Load mask/edge image (ensure it's in grayscale)
imagepath = Constants.IMAGE1_PATH
img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)

# Apply Hough Line Transform
lines = cv2.HoughLinesP(
    img,
    rho=1,                # distance resolution in pixels
    theta=np.pi/180,      # angle resolution in radians
    threshold=80,         # accumulate votes threshold
    minLineLength=200,     # minimum length of line
    maxLineGap=1         # maximum gap between lines
)

vertices = []

if lines is not None:
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        print(f"Line {i+1}: ({x1},{y1}) -> ({x2},{y2})")
        vertices.append((x1, y1))
        vertices.append((x2, y2))
else:
    print("No lines detected.")

plotpoints(imagepath, vertices)