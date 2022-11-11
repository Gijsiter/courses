from keypoint_matching import detect_and_match
from RANSAC import ransac
from transform_image import inverse_warp
import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch(im1, im2, N, P):
    # Get SIFT features.
    ic1, ic2 = detect_and_match(im1, im2, visualize=False)
    # Estimate transform.
    M, _, _ = ransac(ic1, ic2, N, P)
    m, t = M[:4].reshape(2,2), M[4:].reshape(2,1)
    transform = np.vstack((np.hstack((m, t)), np.array([[0,0,1]])))
    # Stitch images.
    im_warped = inverse_warp(im2, transform, im1.shape)
    H, W = im1.shape[:2]
    im_warped[:H, :W, :] = im1

    plt.imshow(im_warped[:,:,::-1])
    plt.show()

    return None


if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    # im1 = cv2.imread('left.jpg')
    # im2 = cv2.imread('right.jpg')

    stitch(im1, im2, 100, 4)
