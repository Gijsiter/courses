import cv2
import numpy as np
import scipy
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt

def divide_regions(img, window_size=15):
    regions = []
    
    for i in range(0, img.shape[0], window_size):
        for j in range(0, img.shape[1], window_size):
            regions.append(img[i:i+window_size, j:j+window_size])
    return regions

def compute_Ab(img1, img2):
    I_x = scipy.signal.convolve2d(img1, np.array([[-1,0,1]]), mode='same')
    I_y = scipy.signal.convolve2d(img1, np.array([[-1],[0],[1]]), mode='same')
    A = np.dstack((I_x.flatten(), I_y.flatten()))[0]

    I_t = img1-img2
    
    b = -1*I_t.flatten()

    return np.linalg.inv(A.T@A)@A.T@b


def optical_flow(img1_path, img2_path):
    # make 2d array from images and float 32
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY).astype(np.float32) /255
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY).astype(np.float32) /255

    regions1 = divide_regions(img1)
    regions2 = divide_regions(img2)

    # Calculate optical flow for each region
    of = []
    for region1, region2 in zip(regions1, regions2):
        of.append(compute_Ab(region1,region2))

    # plot image
    plt.imshow(cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_RGB2BGR))

    z = 0

    X = [i for i in range(0, cv2.imread(img1_path).shape[1],15)]
    Y = [i for i in range(0, cv2.imread(img1_path).shape[0], 15)]
    X_ = []
    Y_ = []
    for y in Y:
        X_.append(X)
        Y_.append(len(X) * [y])

    X_ = np.array(X_).flatten()

    Y_ = np.array(Y_).flatten()

    plt.quiver(X_, Y_, [i[1] for i in of],[i[0] for i in of])

    plt.show()

optical_flow("images/Car1.jpg", "images/Car2.jpg")
