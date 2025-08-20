import cv2
import numpy as np

def find_quadrilaterals(image_path, min_area=1000):
    # Load the image (already binary edges)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # For visualization, convert grayscale to BGR
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Find contours from white edges
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quadrilaterals = []
    for cnt in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Check for 4-sided polygons
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > min_area:
                pts = approx.reshape(4, 2)
                quadrilaterals.append(pts)
                # Draw quadrilateral
                cv2.polylines(img_vis, [pts], isClosed=True, color=(0,255,0), thickness=2)
                # Plot coordinates
                for i, (x, y) in enumerate(pts):
                    cv2.circle(img_vis, (x, y), 6, (0,0,255), -1)
                    cv2.putText(img_vis, f"{x},{y}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Print quadrilateral corners
    for idx, quad in enumerate(quadrilaterals, 1):
        print(f"Quadrilateral {idx}: {quad.tolist()}")

    print(f"Total quadrilaterals detected: {len(quadrilaterals)}")

    # Show the image with plotted coordinates
    cv2.imshow(f"Quadrilaterals in {image_path}", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Change these paths to your edge-processed input images
    image_paths = ['Heatmap/i1.png', 'Heatmap/i2.png', 'Heatmap/i3.png']
    for path in image_paths:
        print(f"\nDetecting in {path}:")
        find_quadrilaterals(path)