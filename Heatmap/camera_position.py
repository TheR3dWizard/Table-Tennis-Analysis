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

    #insert advanced math functions to magically solve everything here
    def detect_table_edges(self, image_path):
        """
        Detects edges in an image of a table tennis table.
        Args:
            image_path (str): Path to the input image.
        Returns:
            edges (numpy.ndarray): Edge-detected image.
        """

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Display the edges on the image
        # cv2.imshow("Edges", edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        size = 7
        kernel = np.ones((size, size), np.uint8)   # adjust kernel size as needed
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # Display the closed on the image
        # cv2.imshow("Closed", closed)
        # cv2.waitKey(0)

        # Apply Hough Line Transform to detect straight edges
        lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=13)
        closed_color = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for colored lines
        if lines is not None:
            lines_img = np.zeros_like(closed_color)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(closed_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  #overlaying for viewing
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  #actual output of lines
            show_image(closed_color, "Hough Lines")
        else:
            print("No straight edges detected.")

        # Find contours
        hough_lines_image = cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(hough_lines_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        show_image(hough_lines_image, "Hough Lines Only")

        # Take the largest contour (table border)
        hough_lines_image = cv2.cvtColor(hough_lines_image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for drawing
        c = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(c, True)  # tweak factor
        c = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(hough_lines_image, [c], 0, (0, 255, 0), 2)  # Draw the largest contour
        show_image(hough_lines_image, "Largest Contour")

        # Get bounding box
        x, y, l, w = cv2.boundingRect(c)

        print(f"Number of vertices: {len(c)}")
        # print(f"Table length (pixels): {l}")
        # print(f"Table width (pixels): {w}")

        # (Optional) Draw rectangle for visualization
        cv2.rectangle(edges, (x, y), (x + l, y + w), (255, 255, 0), 2)
        show_image(edges, "Detected Table")

        return l,w

    
    def find_camera_ratio(self, image_path):
        """
        Calculates the camera ratio based on the table dimensions and detected edges.
        Args:
            image_path (str): Path to the input image.
        Returns:
            camera_ratio (float): Ratio of camera distance to table dimensions.
        """
        l,w = self.detect_table_edges(image_path)     
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

if __name__ == "__main__":
    camera_analysis = CameraAnalysis()
    camera_analysis.find_camera_ratio("akash_001.jpg")
    print(f"Camera Ratio (length): {camera_analysis.camera_ratio_x}")
    print(f"Camera Ratio (width): {camera_analysis.camera_ratio_y}")