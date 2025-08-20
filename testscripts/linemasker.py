import cv2
import numpy as np
from constants import Constants

def frame_to_straight_line_mask(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise (optional but often helps)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Apply morphological closing to close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Create a black mask to draw white lines
    mask = np.zeros_like(closed_edges)
    
    # Detect straight lines using Hough Line Transform
    lines = cv2.HoughLinesP(closed_edges, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=80, 
                            minLineLength=50, 
                            maxLineGap=10)
    
    # Draw straight lines on mask in white
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=2)
    
    return mask
# Example usage:
frame = cv2.imread('assets/specialframes/frame_0197.jpg')   # Or replace with video frame captured via cv2.VideoCapture
mask = frame_to_straight_line_mask(frame)
cv2.imwrite(Constants.IMAGE1_PATH, mask)
