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



    #insert advanced math functions to magically solve everything here
    def detect_table_edges(self, image_path,step=False):
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

    def annotate_corners(self,image,corners):
        """
        Annotates detected corners on the image.
        Args:
            image (numpy.ndarray): Input image.
        Returns:
            annotated_image (numpy.ndarray): Image with annotated corners.
        """
        for corner in corners:
            y, x = corner
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        show_image(image, "Annotated Corners")
        return image

if __name__ == "__main__":
    image_path = "akash_004.jpg"
    camera_analysis = CameraAnalysis()
    bounding_rect = camera_analysis.detect_table_edges(image_path)
    table_crop = camera_analysis.get_cropped_image(image_path, bounding_rect)
    corners = camera_analysis.harris_corners(table_crop)
    camera_analysis.annotate_corners(table_crop, corners)