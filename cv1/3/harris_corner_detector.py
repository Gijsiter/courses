import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, rotate
import matplotlib.pyplot as plt
import cv2


def plot_images(image, Ix, Iy, corner_idx):
    plt.title("First order derivative w.r.t. x")
    plt.imshow(Ix, cmap='gray')
    plt.show()

    plt.title("First order derivative w.r.t. y")
    plt.imshow(Iy, cmap='gray')
    plt.show()

    plt.title("Original image with detected corners")
    plt.imshow(image, cmap='gray')
    ridx, cidx = corner_idx
    plt.scatter(cidx, ridx, c='r', s=5, marker='o')
    plt.show()

    return None

def HCD(image, threshold=0, n=3, visualize=True):
    # Convert color to grayscale if necessary.
    if len(image.shape) == 3:
        im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        im_gray = image
    # Compute image gradients.
    Gd = np.array([[1, 0, -1]])
    Ix = convolve2d(im_gray, Gd, mode='same')
    Iy = convolve2d(im_gray, Gd.T, mode='same')
    # Compute elements of Q.
    A = gaussian_filter(Ix**2, 1, truncate=3)
    B = gaussian_filter(Ix*Iy, 1, truncate=3)
    C = gaussian_filter(Iy**2, 1, truncate=3)
    # Compute H.
    H = (A*C - B**2) - 0.04*(A+C)**2
    # Find corners.
    pad = n // 2
    H_padded = np.pad(H, ((pad, pad), (pad, pad)))
    r, c = [], []
    h, w = H.shape
    for i in range(pad, h + pad):
        for j in range(pad, w + pad):
            region = H_padded[i-pad:i+pad+1, j-pad:j+pad+1]
            max_idx = np.where(region == np.max(region))
            if len(max_idx[0]) == len(max_idx[1]) == 1:
                y, x = max_idx[0][0], max_idx[1][0]
                if y == x == pad:
                    if region[y, x] > threshold:
                        r.append(i-pad)
                        c.append(j-pad)

    if visualize:
        plot_images(image, Ix, Iy, (r, c))

    return H, r, c

if __name__ == '__main__':
    im1 = plt.imread('images/toy/0001.jpg')
    # shape = np.array(img1.shape)
    # center = shape[::-1] / 2
    # im1 = rotate(img1, 45)
    # im2 = rotate(img1, 90)
    im2 = plt.imread('images/doll/0200.jpg')
    threshold = 10000
    n = 3
    HCD(im1, threshold=threshold, n=3)
    HCD(im2, threshold=threshold, n=3)
