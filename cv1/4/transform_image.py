import numpy as np
import cv2
import matplotlib.pyplot as plt


def inverse_warp(im, transform, shape_im2=(0, 0)):
    inv_transform = np.linalg.inv(transform)
    H, W = im.shape[:2]
    # Compute shape for the stiched image.
    corners = np.array([[0, W-1, 0, W-1],
                        [0, 0, H-1, H-1],
                        [1, 1, 1, 1]])
    corners_tr = inv_transform @ corners
    hmin = np.amin(corners_tr[1]) if np.amin(corners_tr[1]) < 0 else 0
    wmin = np.amin(corners_tr[0]) if np.amin(corners_tr[0]) < 0 else 0
    Hn = np.ceil(
        np.ceil(np.amax(corners_tr[1])) + abs(np.floor(hmin))
    ).astype(np.int)
    Wn = np.ceil(
        np.ceil(abs(np.amax(corners_tr[0]))) + abs(np.floor(wmin))
    ).astype(np.int)
    if Hn < shape_im2[0]:
        Hn = shape_im2[0]
    if Wn < shape_im2[1]:
        Wn = shape_im2[1]
    new_image = np.ones((Hn, Wn, 3))

    # Get values for warped image.
    for v in range(Hn):
        for u in range(Wn):
            x, y = (transform@np.array([u, v, 1]))[:2].astype(np.int)
            if 0 <= y < H and 0 <= x < W:
                new_image[v, u,:] = im[y, x,:].astype(np.int)

    return new_image.astype(np.int)
