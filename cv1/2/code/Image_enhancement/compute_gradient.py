import cv2
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def compute_gradient(image_path):
    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    x_dir = [[1,0,-1],[2,0,-2],[1,0,-1]]
    y_dir = [[1,2,1],[0,0,0],[-1,-2,-1]]
    Gx = signal.convolve2d(I, x_dir, boundary='symm')
    Gy = signal.convolve2d(I, y_dir, boundary='symm')

    im_magnitude = np.sqrt(Gx**2 + Gy**2)
    im_direction = np.arctan2(Gy,Gx)

    return Gx, Gy, im_magnitude, im_direction
