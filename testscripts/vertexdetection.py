# import the necessary packages
import numpy as np
import cv2
# import opencv_wrapper as cvw

# import  poly_point_isect as bot

# construct the argument parse and parse the arguments
# load the image
image = cv2.imread("assets/demo/frame_0133.jpg")

# define the list of boundaries
boundaries = [([180, 180, 100], [255, 255, 255])]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

# Start my code
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

corners = cv2.cornerHarris(gray, 9, 3, 0.01)
corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

ret, thresh = cv2.threshold(corners, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(image, (cX, cY), 3, (0, 0, 255), -1)

cv2.imshow("Image", image)
cv2.waitKey(0)