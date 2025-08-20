import cv2
import numpy as np

class CameraAnalysis:
    def __init__(self):
        self.camera_ratio = None

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
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

        size = 7
        kernel = np.ones((size, size), np.uint8)   # adjust kernel size as needed
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # Display the closed on the image
        cv2.imshow("Closed", closed)
        cv2.waitKey(0)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Take the largest contour (table border)
        c = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, l, w = cv2.boundingRect(c)

        print(f"Table length (pixels): {l}")
        print(f"Table width (pixels): {w}")

        # (Optional) Draw rectangle for visualization
        cv2.rectangle(edges, (x, y), (x + l, y + w), (255, 255, 0), 2)
        cv2.imshow("Detected Table", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        

if __name__ == "__main__":
    camera_analysis = CameraAnalysis()
    edges = camera_analysis.detect_table_edges("test.png")
    # You can save or further process the edges as needed
    # cv2.imwrite("edges_output.jpg", edges)