
#!/usr/bin/env python3
"""
Table Tennis Table Detection Program
=====================================

This program detects table tennis table vertices and calculates the midpoint
using multiple computer vision techniques when YOLO fails to recognize the table.

Author: AI Assistant
Requirements: opencv-python, numpy

Usage:
    python table_tennis_detector.py <image_path>

Example:
    python table_tennis_detector.py table_tennis_frame.jpg
"""

import cv2
import numpy as np
import math
import sys
import os
from typing import List, Tuple, Optional


class TableTennisTableDetector:
    """
    A comprehensive class for detecting table tennis table vertices and calculating midpoint.
    Implements multiple detection strategies since YOLO failed to recognize the table.
    """
    
    def __init__(self):
        # Standard table tennis table dimensions (in cm)
        self.standard_width = 274  # cm
        self.standard_height = 152.5  # cm
        self.standard_ratio = self.standard_width / self.standard_height  # ~1.8
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better table detection
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_table_edges(self, image: np.ndarray, low_threshold: int = 50, 
                          high_threshold: int = 150) -> np.ndarray:
        """
        Detect edges using Canny edge detection with optimized parameters for table tennis tables
        """
        preprocessed = self.preprocess_image(image)
        
        # Apply Canny edge detection
        edges = cv2.Canny(preprocessed, low_threshold, high_threshold, apertureSize=3, L2gradient=True)
        
        # Apply morphological operations to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_table_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find table contours using contour detection and approximation
        """
        edges = self.detect_table_edges(image)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        table_candidates = []
        
        for contour in contours:
            # Get contour area
            area = cv2.contourArea(contour)
            
            # Filter by minimum area (table should be substantial in the image)
            min_area = (image.shape[0] * image.shape[1]) * 0.1  # At least 10% of image
            if area < min_area:
                continue
                
            # Approximate contour to reduce vertices
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if approximated contour has 4 vertices (rectangle-like)
            if len(approx) == 4:
                # Check aspect ratio
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if height > 0:
                    aspect_ratio = width / height
                    # Table tennis table has aspect ratio around 1.8
                    if 1.3 < aspect_ratio < 2.5 or 1.3 < (1/aspect_ratio) < 2.5:
                        table_candidates.append(approx)
        
        return table_candidates
    
    def detect_lines_hough(self, image: np.ndarray) -> Tuple[List, List]:
        """
        Detect table lines using Hough Transform
        """
        edges = self.detect_table_edges(image)
        
        # Apply probabilistic Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=50)
        
        if lines is None:
            return [], []
        
        # Filter and group lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            if x2 != x1:
                angle = math.atan2(abs(y2 - y1), abs(x2 - x1)) * 180 / math.pi
            else:
                angle = 90
                
            # Classify as horizontal or vertical
            if angle < 20:  # Nearly horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif angle > 70:  # Nearly vertical
                vertical_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines, vertical_lines
    
    def find_line_intersections(self, horizontal_lines: List, vertical_lines: List) -> List[Tuple[int, int]]:
        """
        Find intersection points between horizontal and vertical lines
        """
        intersections = []
        
        for h_line in horizontal_lines:
            hx1, hy1, hx2, hy2 = h_line
            for v_line in vertical_lines:
                vx1, vy1, vx2, vy2 = v_line
                
                # Calculate intersection
                intersection = self.line_intersection(
                    (hx1, hy1), (hx2, hy2),
                    (vx1, vy1), (vx2, vy2)
                )
                
                if intersection:
                    intersections.append(intersection)
        
        return intersections
    
    def line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        """
        Find intersection point of two lines
        """
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        
        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)
        
        return (int(x), int(y))
    
    def detect_harris_corners(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect corners using Harris corner detection as backup method
        """
        preprocessed = self.preprocess_image(image)
        
        # Harris corner detection
        corners = cv2.cornerHarris(preprocessed, blockSize=2, ksize=3, k=0.04)
        
        # Dilate corner image to enhance corner points
        corners = cv2.dilate(corners, None)
        
        # Threshold for optimal corner detection
        threshold = 0.01 * corners.max()
        
        # Find corner coordinates
        corner_coords = []
        corner_locations = np.where(corners > threshold)
        
        for y, x in zip(corner_locations[0], corner_locations[1]):
            corner_coords.append((x, y))
        
        return corner_coords
    
    def filter_table_corners(self, corners: List[Tuple[int, int]], 
                           image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Filter and select the best 4 corners that likely represent table corners
        """
        if len(corners) < 4:
            return corners
        
        # Convert to numpy array for easier processing
        corners_array = np.array(corners)
        
        # Find convex hull to get outermost points
        hull = cv2.convexHull(corners_array.astype(np.float32))
        hull_points = [(int(point[0][0]), int(point[0][1])) for point in hull]
        
        # If we have exactly 4 points, return them
        if len(hull_points) == 4:
            return hull_points
        
        # Otherwise, try to find 4 corners by clustering or geometric analysis
        return self.select_best_four_corners(hull_points, image_shape)
    
    def select_best_four_corners(self, corners: List[Tuple[int, int]], 
                               image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Select the best 4 corners from a larger set of corner candidates
        """
        if len(corners) <= 4:
            return corners
        
        h, w = image_shape[:2]
        
        # Find corners closest to image corners
        image_corners = [(0, 0), (w, 0), (w, h), (0, h)]
        selected_corners = []
        
        for img_corner in image_corners:
            # Find the corner candidate closest to this image corner
            min_dist = float('inf')
            best_corner = None
            
            for corner in corners:
                dist = np.sqrt((corner[0] - img_corner[0])**2 + (corner[1] - img_corner[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_corner = corner
            
            if best_corner and best_corner not in selected_corners:
                selected_corners.append(best_corner)
        
        # If we still don't have 4 corners, add remaining ones
        while len(selected_corners) < 4 and len(corners) > len(selected_corners):
            for corner in corners:
                if corner not in selected_corners:
                    selected_corners.append(corner)
                    break
        
        return selected_corners[:4]
    
    def order_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Order corners in consistent manner: top-left, top-right, bottom-right, bottom-left
        """
        corners_array = np.array(corners)
        
        # Calculate center point
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])
        
        # Sort corners by angle from center
        def angle_from_center(corner):
            return math.atan2(corner[1] - center_y, corner[0] - center_x)
        
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # Ensure we start from top-left quadrant
        # Find the corner with minimum sum of coordinates (top-left)
        sums = [x + y for x, y in sorted_corners]
        min_sum_idx = sums.index(min(sums))
        
        # Rotate the list to start with top-left corner
        ordered_corners = sorted_corners[min_sum_idx:] + sorted_corners[:min_sum_idx]
        
        return ordered_corners
    
    def calculate_midpoint(self, corners: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        Calculate the midpoint of the table from its corners
        """
        if not corners:
            return None
        
        corners_array = np.array(corners)
        center_x = np.mean(corners_array[:, 0])
        center_y = np.mean(corners_array[:, 1])
        
        return (float(center_x), float(center_y))
    
    def apply_perspective_correction(self, image: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply perspective correction to get a top-down view of the table
        """
        if len(corners) != 4:
            return image
        
        # Order corners properly
        ordered_corners = self.order_corners(corners)
        
        # Define source points (detected corners)
        src_points = np.float32(ordered_corners)
        
        # Calculate dimensions for the corrected view
        # Use standard table tennis table ratio
        width = 600
        height = int(width / self.standard_ratio)
        
        # Define destination points (rectangular view)
        dst_points = np.float32([
            [0, 0],              # top-left
            [width, 0],          # top-right
            [width, height],     # bottom-right
            [0, height]          # bottom-left
        ])
        
        # Get perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective warp
        corrected = cv2.warpPerspective(image, matrix, (width, height))
        
        return corrected
    
    def detect_table_vertices(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], Tuple[float, float]]:
        """
        Main method to detect table tennis table vertices and calculate midpoint
        """
        print("Starting table tennis table detection...")
        
        # Method 1: Contour-based detection
        print("Attempting contour-based detection...")
        table_contours = self.find_table_contours(image)
        
        if table_contours:
            # Use the largest contour (most likely the table)
            largest_contour = max(table_contours, key=cv2.contourArea)
            corners = [(point[0][0], point[0][1]) for point in largest_contour]
            corners = self.order_corners(corners)
            midpoint = self.calculate_midpoint(corners)
            
            print(f"Contour method found {len(corners)} corners")
            return corners, midpoint
        
        # Method 2: Hough line detection
        print("Attempting Hough line detection...")
        try:
            horizontal_lines, vertical_lines = self.detect_lines_hough(image)
            
            if horizontal_lines and vertical_lines:
                intersections = self.find_line_intersections(horizontal_lines, vertical_lines)
                
                if len(intersections) >= 4:
                    # Filter intersections to get table corners
                    corners = self.filter_table_corners(intersections, image.shape)
                    corners = self.order_corners(corners)
                    midpoint = self.calculate_midpoint(corners)
                    
                    print(f"Hough line method found {len(corners)} corners")
                    return corners, midpoint
        except Exception as e:
            print(f"Hough line method failed: {e}")
        
        # Method 3: Harris corner detection
        print("Attempting Harris corner detection...")
        try:
            harris_corners = self.detect_harris_corners(image)
            
            if len(harris_corners) >= 4:
                corners = self.filter_table_corners(harris_corners, image.shape)
                corners = self.order_corners(corners)
                midpoint = self.calculate_midpoint(corners)
                
                print(f"Harris corner method found {len(corners)} corners")
                return corners, midpoint
        except Exception as e:
            print(f"Harris corner method failed: {e}")
        
        print("No table detected with current methods")
        return [], None
    
    def visualize_detection(self, image: np.ndarray, corners: List[Tuple[int, int]], 
                          midpoint: Tuple[float, float]) -> np.ndarray:
        """
        Visualize the detected table vertices and midpoint
        """
        result_img = image.copy()
        
        if corners:
            # Draw corners
            for i, corner in enumerate(corners):
                cv2.circle(result_img, corner, 10, (0, 255, 0), -1)
                cv2.putText(result_img, f"C{i+1}", (corner[0]+15, corner[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw table outline
            corners_array = np.array(corners, dtype=np.int32)
            cv2.polylines(result_img, [corners_array], True, (255, 0, 0), 3)
        
        if midpoint:
            # Draw midpoint
            midpoint_int = (int(midpoint[0]), int(midpoint[1]))
            cv2.circle(result_img, midpoint_int, 15, (0, 0, 255), -1)
            cv2.putText(result_img, "Midpoint", (midpoint_int[0]+20, midpoint_int[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result_img


# Alternative detection methods for better robustness
def detect_green_table(image):
    """
    Detect table tennis table based on green color filtering
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color (table tennis tables are often green)
    # Adjust these ranges based on your specific table color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest rectangular contour
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            # Ensure it's large enough to be a table
            if area > 10000:  # Adjust threshold as needed
                return approx
    
    return None


def detect_table_adaptive_edges(image):
    """
    Use adaptive edge detection with multiple threshold values
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    best_corners = []
    best_score = 0
    
    # Try different Canny thresholds
    threshold_pairs = [(50, 150), (30, 100), (80, 200), (20, 80)]
    
    for low_thresh, high_thresh in threshold_pairs:
        # Edge detection
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 5000:  # Too small
                continue
                
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                # Calculate rectangularity score
                rect = cv2.minAreaRect(contour)
                rect_area = rect[1][0] * rect[1][1]
                rectangularity = area / rect_area if rect_area > 0 else 0
                
                # Calculate aspect ratio score (table tennis table is ~1.8:1)
                width, height = rect[1]
                if height > 0:
                    aspect_ratio = width / height
                    aspect_score = 1 - abs(aspect_ratio - 1.8) / 1.8
                else:
                    aspect_score = 0
                
                # Combined score
                score = rectangularity * 0.5 + aspect_score * 0.5
                
                if score > best_score:
                    best_score = score
                    best_corners = [(point[0][0], point[0][1]) for point in approx]
    
    return best_corners if best_score > 0.3 else []


def comprehensive_table_detection(image):
    """
    Try multiple detection methods and return the best result
    """
    results = []
    
    # Method 1: Original comprehensive detector
    detector = TableTennisTableDetector()
    vertices1, midpoint1 = detector.detect_table_vertices(image)
    if vertices1:
        results.append(("Comprehensive", vertices1, midpoint1))
    
    # Method 2: Green color detection
    try:
        green_corners = detect_green_table(image)
        if green_corners:
            corners = [(point[0][0], point[0][1]) for point in green_corners]
            midpoint = ((corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2)
            results.append(("Green Detection", corners, midpoint))
    except:
        pass
    
    # Method 3: Adaptive edges
    try:
        adaptive_corners = detect_table_adaptive_edges(image)
        if adaptive_corners:
            corners_array = np.array(adaptive_corners)
            midpoint = (np.mean(corners_array[:, 0]), np.mean(corners_array[:, 1]))
            results.append(("Adaptive Edges", adaptive_corners, midpoint))
    except:
        pass
    
    # Return the result with the most confidence (or first successful one)
    if results:
        return results[0]  # Return first successful detection
    
    return None, [], None


def main():
    """
    Main function to run table tennis table detection
    """
    
    image_path = "assets/bb/table1.png"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image!")
        return
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Try comprehensive detection
    method, vertices, midpoint = comprehensive_table_detection(image)
    
    # Print results
    print("\n" + "="*50)
    print("DETECTION RESULTS")
    print("="*50)
    
    if vertices and midpoint:
        print(f"Table detected using: {method}")
        print(f"Table vertices detected: {len(vertices)} corners")
        for i, vertex in enumerate(vertices):
            print(f"  Corner {i+1}: ({vertex[0]}, {vertex[1]})")
        
        print(f"\nTable midpoint: ({midpoint[0]:.1f}, {midpoint[1]:.1f})")
        
        # Create detector instance for visualization
        detector = TableTennisTableDetector()
        
        # Visualize results
        result_image = detector.visualize_detection(image, vertices, midpoint)
        
        # Save result
        output_path = "table_detection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"\nResult saved to: {output_path}")
        
        # Apply perspective correction if possible
        if len(vertices) == 4:
            try:
                corrected_image = detector.apply_perspective_correction(image, vertices)
                corrected_output_path = "table_perspective_corrected.jpg"
                cv2.imwrite(corrected_output_path, corrected_image)
                print(f"Perspective corrected image saved to: {corrected_output_path}")
                
                # Calculate midpoint in corrected image
                corrected_midpoint = (corrected_image.shape[1] / 2, corrected_image.shape[0] / 2)
                print(f"Midpoint in corrected view: ({corrected_midpoint[0]:.1f}, {corrected_midpoint[1]:.1f})")
            except Exception as e:
                print(f"Perspective correction failed: {e}")
    
    else:
        print("No table detected!")
        print("\nTips for better detection:")
        print("1. Ensure good lighting and contrast between table and background")
        print("2. Make sure table edges are clearly visible")
        print("3. Try adjusting camera angle for better view of table corners")
        print("4. Ensure table occupies a significant portion of the image")
        print("5. For green tables, ensure proper color balance in the image")


if __name__ == "__main__":
    main()
