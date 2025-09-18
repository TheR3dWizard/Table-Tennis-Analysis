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
        size = 10
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
    
    def all_contours(self,image,step=False,return_image=False):
        """
        Finds all contours in a binary image.
        Args:
            image (numpy.ndarray): Binary image.
        Returns:
            contours (list): List of all contours found.
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if step:
            contour_img = np.zeros_like(image)
            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for drawing
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)  # Draw all contours
            show_image(contour_img, "All Contours")
        if return_image:
            return contours, contour_img
        return contours

    def largest_contour(self,image,step=False):
        """
        Finds the largest contour in a binary image.
        Args:
            image (numpy.ndarray): Binary image.
        Returns:
            largest_contour (numpy.ndarray): The largest contour found.
        """
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # tweak factor
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        if step:
            contour_img = np.zeros_like(image)
            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for drawing
            cv2.drawContours(contour_img, [largest_contour], 0, (0, 255, 0), 2)  # Draw the largest contour
            show_image(contour_img, "Largest Contour")
        return largest_contour

    def get_red_mask(self, image, show=False):
        """
        Returns a mask isolating the red regions in the image.
        Args:
            image (numpy.ndarray): Input BGR image.
            show (bool): If True, displays the mask.
        Returns:
            mask (numpy.ndarray): Binary mask of red regions.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Red can wrap around the hue, so use two ranges
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        if show:
            show_image(mask, "Red Mask")
        return mask
    
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

    def harris_corners(self,image,channels=3):
        """
        Detects Harris corners in the image.
        Args:
            image (numpy.ndarray): Input image.
        Returns:
            corners (list): List of detected corner points.
        """
        if channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
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
    
    def straighten_contour(self, contour):
        """
        Approximates a contour with straight lines (polygonal curve).
        Args:
            contour (numpy.ndarray): Input contour.
        Returns:
            straight_contour (numpy.ndarray): Approximated contour with straight lines.
        """
        epsilon = 0.02 * cv2.arcLength(contour, True)
        straight_contour = cv2.approxPolyDP(contour, epsilon, True)
        return straight_contour
    
    def cvt_contour_to_image(self, contour, image_shape):
        """
        Converts a contour to a binary image.
        Args:
            contour (numpy.ndarray): Input contour.
            image_shape (tuple): Shape of the output image (height, width).
        Returns:
            contour_image (numpy.ndarray): Binary image with the contour drawn.
        """
        contour_image = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(contour_image, [contour], -1, (255), thickness=cv2.FILLED)
        return contour_image
    
    def find_centroid(self, contour):
        """
        Finds the centroid of a contour.
        Args:
            contour (numpy.ndarray): Input contour.
        Returns:
            centroid (tuple): (x, y) coordinates of the centroid.
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    def find_convex_hull_contour(self, contour):
        """
        Finds the convex hull of a contour.
        Args:
            contour (numpy.ndarray): Input contour (N, 1, 2) or (N, 2).
        Returns:
            hull (numpy.ndarray): Convex hull points.
        """
        contour = np.array(contour)
        if len(contour.shape) == 2:
            contour = contour.reshape(-1, 1, 2)
        hull = cv2.convexHull(contour)
        return hull
    
    def extract_table_corners(self,mask):
        """
        Extracts the four corners of the table from the image.
        Args:
            image (numpy.ndarray): Input image.
        Returns:
            corners (list): List of corner points.
        """
        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

        # Red color range (two ranges because red wraps around 0° in HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Binary mask for the table
        table_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Take largest contour (the table)
        c = max(contours, key=cv2.contourArea)

        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if len(approx) != 4:
            # fallback: fit a rectangle if not exactly 4 points
            rect = cv2.minAreaRect(c)
            approx = cv2.boxPoints(rect)

        corners = np.intp(approx).reshape(-1, 2)

        # Draw on image for visualization
        output = mask.copy()
        for (x, y) in corners:
            cv2.circle(output, (x, y), 5, (255, 255, 255), -1)

        cv2.imshow("Table Corners", cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Corners:", corners)


if __name__ == "__main__":
    camera_analysis = CameraAnalysis()
    image_path = f"../images/15.png"
    image = cv2.imread(image_path)
    # table = camera_analysis.get_red_mask(cv2.imread(image_path),show=True)
    # table_contour = camera_analysis.all_contours(table,step=True)[0]
    # # og_contour_image = camera_analysis.cvt_contour_to_image(table_contour[0], table.shape)
    # straight_contour = camera_analysis.straighten_contour(table_contour)
    # contour_image = camera_analysis.cvt_contour_to_image(straight_contour, table.shape)
    # show_image(contour_image,"Straight Contour Image")
    # contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    # corners = camera_analysis.harris_corners(contour_image,channels=3)
    # camera_analysis.annotate_points(contour_image,corners,title="Corners on Cont ours")
    # convex_hull = camera_analysis.find_convex_hull_contour(straight_contour)

    # camera_analysis.extract_table_corners(image)
    camera_analysis.detect_table_boundary(image_path,step=True)




