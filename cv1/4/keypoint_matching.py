import cv2
import matplotlib.pyplot as plt
import numpy as np

# Source for most of the code:
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
def detect_and_match(im1, im2, visualize=False):
    im1gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    SIFT = cv2.SIFT_create()
    kp1, des1 = SIFT.detectAndCompute(im1gray, None)
    kp2, des2 = SIFT.detectAndCompute(im2gray, None)

    matcher = cv2.BFMatcher()
    matches = matcher.match(des1,des2)

    if visualize:
        idx = np.random.randint(0, len(matches), 10)
        img3 = cv2.drawMatches(im1gray, kp1, im2gray, kp2, np.array(matches)[idx],
                               None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()

    # Collect match coordinates.
    im1_coords, im2_coords = [], []
    for mat in matches:
        im1idx = mat.queryIdx
        im2idx = mat.trainIdx
        im1_coords.append(list(kp1[im1idx].pt))
        im2_coords.append(list(kp2[im2idx].pt))

    return np.array(im1_coords), np.array(im2_coords)

if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    detect_and_match(im1, im2)