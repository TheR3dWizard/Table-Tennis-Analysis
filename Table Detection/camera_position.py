import cv2
from matplotlib import contour
import numpy as np

def show_image(image, title="Image"):
    """
    Displays an image in a window.
    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class CameraAnalysis:
    def __init__(self):
        self.camera_ratio_x = None
        self.camera_ratio_y = None
        self.camera_ratio_z = None

        #table dimensions in cm
        self.table_length = 274
        self.table_width = 152.5
        self.table_height = 76


    def edge_detection(self,image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        return edges


    def hough_line_detection(self, edges,step=False):
        # Apply Hough Line Transform to detect straight edges
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=13)
        if step:
            if lines is not None:
                lines_img = np.zeros_like(edges)
                # Convert to 3-channel image for colored lines
                lines_img_color = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2BGR)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(lines_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                show_image(lines_img_color, "Hough Lines")
        return lines

    #insert advanced math functions to magically solve everything here
    def detect_table_boundary(self, image_path,step=False):
        """
        Detects edges in an image of a table tennis table
        Args:
            image_path (str): Path to the input image.
        Returns:
            edges (numpy.ndarray): Edge-detected image.
        """

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read.")
        
        
        edges = self.edge_detection(image)

        # Smooth edges
        size = 7
        kernel = np.ones((size, size), np.uint8)   # adjust kernel size as needed
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


        # Apply Hough Line Transform to detect straight edges
        lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=13)
        closed_color = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for colored lines
        if lines is not None:
            lines_img = np.zeros_like(closed_color)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(closed_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  #overlaying for viewing
                cv2.line(lines_img,    (x1, y1), (x2, y2), (0, 255, 0), 2)  #actual output of lines
            show_image(closed_color, "Hough Lines") if step else None
        else:
            print("No straight edges detected.")

        # Find contours
        hough_lines_image = cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(hough_lines_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        show_image(hough_lines_image, "Hough Lines Only") if step else None

        # Take the largest contour (table border)
        hough_lines_image = cv2.cvtColor(hough_lines_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for drawing
        c = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(c, True)  # tweak factor
        c = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(hough_lines_image, [c], 0, (0, 255, 0), 2)  # Draw the largest contour
        show_image(hough_lines_image, "Largest Contour") if step else None
        
        # Get bounding box
        x, y, l, w = cv2.boundingRect(c)

        print(f"Number of vertices: {len(c)}")
        # print(f"Table length (pixels): {l}")
        # print(f"Table width (pixels): {w}")

        # (Optional) Draw rectangle for visualization   
        cv2.rectangle(edges, (x, y), (x + l, y + w), (255, 255, 0), 2)
        show_image(edges, "Detected Table")

        return x,y,l,w

    
    def find_camera_ratio(self, image_path):
        """
        Calculates the camera ratio based on the table dimensions and detected edges.
        Args:
            image_path (str): Path to the input image.
        Returns:
            camera_ratio (float): Ratio of camera distance to table dimensions.
        """
        l,w = self.detect_table_boundary(image_path)     
        #map pixel dimensions to real-world dimensions
        self.camera_ratio_x = self.table_length / l
        self.camera_ratio_y = self.table_width / w

    def find_absolute_position(self,x,y,image_path):
        """
        Calculates the absolute position of a point in the real world based on camera ratios.
        Args:
            x (float): x-coordinate in pixels.
            y (float): y-coordinate in pixels.
            image_path (str): Path to the input image.
        Returns:
            absolute_position (tuple): (x, y) in cm.
        """
        if self.camera_ratio_x is None or self.camera_ratio_y is None:
            self.find_camera_ratio(image_path)

        absolute_x = x * self.camera_ratio_x
        absolute_y = y * self.camera_ratio_y

        return absolute_x, absolute_y

    def get_cropped_image(self,image_path,bounding_rect):
        """
        Crops the image to the given bounding rectangle.
        Args:
            image_path (str): Path to the input image.
            bounding_rect (tuple): (x, y, width, height) of the bounding rectangle.
        Returns:
            cropped_image (numpy.ndarray): Cropped image.
        """
        image = cv2.imread(image_path)
        x, y, w, h = bounding_rect
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image

    def harris_corners(self,image):
        """
        Detects Harris corners in the image.
        Args:
            image (numpy.ndarray): Input image.
        Returns:
            corners (list): List of detected corner points.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        corners = np.argwhere(dst > 0.01 * dst.max())
        return corners

    def annotate_points(self,image,points,color=(0,0,255),title="Annotated Points"):
        """
        Annotates detected points on the image.
        Args:
            image (numpy.ndarray): Input image.
        Returns:
            annotated_image (numpy.ndarray): Image with annotated points.
        """
        image_copy = image.copy()
        for point in points:
            y, x = point
            cv2.circle(image_copy, (x, y), 1, color, -1)
        show_image(image_copy, title)
        return image_copy
    
    def get_intersections_between_points_and_line(self,points,line):
        """
        Finds intersection points between a set of points and a line.
        Args:
            points (list): List of points (x, y).
            line (tuple): Line defined by two points ((x1, y1), (x2, y2)).
        Returns:
            intersections (list): List of intersection points.
        """
        (x1, y1), (x2, y2) = line[0].reshape(2, 2)
        intersections = []
        for (px, py) in points:
            # Check if point is on the line segment
            if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
                # Calculate the area of the triangle formed by the line and the point
                area = abs((x2 - x1) * (py - y1) - (px - x1) * (y2 - y1))
                if area < 10000000000000000:  # Threshold to consider as intersection
                    intersections.append((px, py))
        return intersections
    
    def get_intersections_between_point_and_lines(self,points,lines,image=None):
        """
        Finds intersection points between a set of points and multiple lines.
        Args:
            points (list): List of points (x, y).
            lines (list): List of lines, each defined by two points ((x1, y1), (x2, y2)).
        Returns:
            intersections (list): List of intersection points.
        """
        intersections = []
        for line in lines:
            print("Line is",line)
            line_intersections = self.get_intersections_between_points_and_line(points, line)
            intersections.extend(line_intersections)
        if image is not None:
            self.annotate_points(image, intersections, color=(255, 0, 0), title="Intersections")
        return intersections
    
    def intersect_table_edges_and_corners(self,image_path):
        """
        Detects table edges and Harris corners, then finds their intersections.
        Args:
            image_path (str): Path to the input image.
        Returns:
            intersections (list): List of intersection points.
        """
        bounding_rect = self.detect_table_boundary(image_path)
        table_crop = self.get_cropped_image(image_path, bounding_rect)
        points = self.harris_corners(table_crop)
        edges = self.edge_detection(table_crop)
        edges = self.hough_line_detection(edges)
        intersections = self.get_intersections_between_point_and_lines(points, edges,image=table_crop)
        return intersections

if __name__ == "__main__":
    camera_analysis = CameraAnalysis()
    for i in range(1,5):
        image_path = f"akash_00{i}.jpg"
        intersections = camera_analysis.intersect_table_edges_and_corners(image_path)
    