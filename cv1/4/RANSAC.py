import warnings
import numpy as np
import cv2
from keypoint_matching import detect_and_match
from numpy.linalg.linalg import inv
from transform_image import inverse_warp
import matplotlib.pyplot as plt


def construct_A(im_idx):
    A = np.concatenate(
        [np.array([[x, y, 0, 0, 1, 0],
                   [0, 0, x, y, 0, 1]]) for x,y in im_idx], axis=0
    )
    return A


def solve_M(im1_idx, im2_idx):
    A = construct_A(im1_idx)
    b = im2_idx.reshape(-1,1)

    return np.linalg.inv(A.T@A)@A.T@b


def ransac(imc_1, imc_2, N=1000, P=4):
    most_inliers = 0
    transform = None
    inliers_best = None
    inlier_targets = None
    for i in range(N):
        samp_idx = np.random.randint(0, imc_1.shape[0], P)
        samples1, samples2 = imc_1[samp_idx], imc_2[samp_idx]
        pos_transform = solve_M(samples1, samples2)
        A = construct_A(imc_1)
        b_approx = (A@pos_transform).reshape(-1,2)
        inliers = imc_1[abs(np.linalg.norm(b_approx - imc_2, axis=1)) < 10]
        if inliers.shape[0] > most_inliers:
            transform = pos_transform.copy()
            inliers_best = inliers.copy()
            inlier_targets = imc_2[
                               np.linalg.norm(b_approx - imc_2, axis=1) < 10
                            ].copy()
            most_inliers = inliers.shape[0]

    transform = solve_M(inliers_best, inlier_targets)

    return transform, inliers_best, inlier_targets


if __name__ == '__main__':
    im1 = cv2.imread('boat1.pgm')
    im2 = cv2.imread('boat2.pgm')
    # Compute keypoints.
    imc_1, imc_2 = detect_and_match(im1, im2)

    # Create homogeneous transformation matrix.
    M, _, _ = ransac(imc_1, imc_2, N=100)
    m, t = M[:4].reshape(2,2), M[4:].reshape(2,1)
    transform = np.vstack((np.hstack((m, t)), np.array([[0,0,1]])))

    # Warp and plot images
    im1_tr = inverse_warp(im1, np.linalg.inv(transform))
    im2_tr = inverse_warp(im2, transform)
    fig, ax = plt.subplots(1,2)
    ax[0].set_title("Image 1 to image 2")
    ax[0].imshow(im1_tr[:,:,::-1])
    ax[1].set_title("Image 2 to image 1")
    ax[1].imshow(im2_tr[:,:,::-1])
    plt.show()
