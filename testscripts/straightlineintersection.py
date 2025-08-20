import cv2
import numpy as np
from constants import Constants
from helper import plotpoints

def line_to_params(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = A*x1 + B*y1
    return A, B, C

def compute_intersection(line1, line2):
    A1, B1, C1 = line_to_params(*line1)
    A2, B2, C2 = line_to_params(*line2)
    determinant = A1*B2 - A2*B1
    if determinant == 0:
        return None
    x = (B2*C1 - B1*C2) / determinant
    y = (A1*C2 - A2*C1) / determinant
    return int(round(x)), int(round(y))

def is_point_on_segment(x, y, x1, y1, x2, y2, tol=2):
    min_x, max_x = min(x1, x2) - tol, max(x1, x2) + tol
    min_y, max_y = min(y1, y2) - tol, max(y1, y2) + tol
    return min_x <= x <= max_x and min_y <= y <= max_y

def get_line_segments(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
    return [line[0] for line in lines] if lines is not None else []

def filter_points_by_distance(points, min_dist):
    filtered = []
    for pt in points:
        if all(np.hypot(pt[0] - fp[0], pt[1] - fp[1]) >= min_dist for fp in filtered):
            filtered.append(pt)
    return filtered

def find_intersections(img_path, min_dist=20):
    lines = get_line_segments(img_path)
    points = []
    n = len(lines)
    for i in range(n):
        for j in range(i+1, n):
            l1 = lines[i]
            l2 = lines[j]
            pt = compute_intersection(l1, l2)
            if pt is not None:
                x, y = pt
                if is_point_on_segment(x, y, *l1) and is_point_on_segment(x, y, *l2):
                    points.append(pt)
    # Now filter points by minimum distance
    points = filter_points_by_distance(points, min_dist)
    return points

def draw_points_on_mask(img_path, points, radius=5):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros_like(img)
    for (x, y) in points:
        cv2.circle(mask, (x, y), radius, 255, thickness=-1)
    return mask


# Usage
img_path = Constants.IMAGE1_PATH   # Use your mask image filename here
min_distance = 30         # Minimum distance between points in pixels (adjustable)
intersections = find_intersections(img_path, min_dist=min_distance)
mask = draw_points_on_mask(img_path, intersections)
cv2.imwrite('intersections_mask_filtered.jpeg', mask)
plotpoints(img_path, list(intersections))
print(f"Found {len(intersections)} intersection points (with minimum distance={min_distance}px). Mask saved as 'intersections_mask_filtered.jpeg'")
