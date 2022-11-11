from gauss2D import gauss2D
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_LoG(image, LOG_type):
    if LOG_type == 1:
        gauss = gauss2D(0.5, 0.5, 5)
        img_smooth = convolve2d(image, gauss, mode='same')
        laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]])
        imOut = convolve2d(img_smooth, laplacian, mode='same')

    elif LOG_type == 2:
        gauss = gauss2D(0.5, 0.5 ,5)
        laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]])
        LoG = convolve2d(gauss, laplacian, mode='same')
        imOut = convolve2d(image, LoG, mode='same')

    elif LOG_type == 3:
        k1 = gauss2D(0.5, 0.5, 5)
        k2 = gauss2D(1.6*0.5, 1.6*0.5, 5)
        LoG = k1 - k2
        imOut = convolve2d(image, LoG, mode='same')

    # Normalize image.
    imOut = (imOut - imOut.min())*(1/(imOut.max()-imOut.min()))

    return imOut

if __name__ == '__main__':
    img = cv2.imread('images/image2.jpg', )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = [compute_LoG(img, i) for i in range(1,4)]
    for i, r in enumerate(res):
        plt.title(f'Result after applying method {i+1}')
        plt.imshow(r, cmap='gray')
        plt.show()