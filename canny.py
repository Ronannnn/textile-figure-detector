import cv2 as cv
import numpy as np


# Reference: https://www.geeksforgeeks.org/implement-canny-edge-detector-in-python-using-opencv/
def canny_detector(img, thresh1, thresh2):

    # Noise reduction step
    img = cv.GaussianBlur(img, (5, 5), 1.4)

    # Calculating the gradients
    gx = cv.Sobel(np.float32(img), cv.CV_64F, 1, 0, 3)
    gy = cv.Sobel(np.float32(img), cv.CV_64F, 0, 1, 3)

    # Conversion of Cartesian coordinates to polar
    mag, ang = cv.cartToPolar(gx, gy, angleInDegrees=True)

    # setting the minimum and maximum thresholds
    # for double thresholding
    mag_max = np.max(mag)
    if not thresh1:
        thresh1 = mag_max * 0.1
    if not thresh2:
        thresh2 = mag_max * 0.5

    # getting the dimensions of the input image
    height, width = img.shape

    # Looping through every pixel of the grayscale
    # image
    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbors of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang <= 22.5:
                neighbor_1_x, neighbor_1_y = i_x-1, i_y
                neighbor_2_x, neighbor_2_y = i_x + 1, i_y

            # top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighbor_1_x, neighbor_1_y = i_x-1, i_y-1
                neighbor_2_x, neighbor_2_y = i_x + 1, i_y + 1

            # In y-axis direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighbor_1_x, neighbor_1_y = i_x, i_y-1
                neighbor_2_x, neighbor_2_y = i_x, i_y + 1

            # top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighbor_1_x, neighbor_1_y = i_x-1, i_y + 1
                neighbor_2_x, neighbor_2_y = i_x + 1, i_y-1

            # Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighbor_1_x, neighbor_1_y = i_x-1, i_y
                neighbor_2_x, neighbor_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            if width > neighbor_1_x >= 0 and height > neighbor_1_y >= 0:
                if mag[i_y, i_x] < mag[neighbor_1_y, neighbor_1_x]:
                    mag[i_y, i_x] = 0
                    continue

            if width > neighbor_2_x >= 0 and height > neighbor_2_y >= 0:
                if mag[i_y, i_x] < mag[neighbor_2_y, neighbor_2_x]:
                    mag[i_y, i_x] = 0

    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)

    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):

            grad_mag = mag[i_y, i_x]

            if grad_mag < thresh1:
                mag[i_y, i_x] = 0
            elif thresh2 > grad_mag >= thresh1:
                ids[i_y, i_x] = 1
            else:
                ids[i_y, i_x] = 2

    # finally returning the magnitude of
    # gradients of edges
    mag = mag.astype(np.uint8)
    return mag
