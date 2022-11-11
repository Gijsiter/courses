import numpy as np
import cv2
from numpy.core.defchararray import array
from numpy.lib.function_base import append
import getColourChannels

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods
    R, G, B = getColourChannels.getColourChannels(input_image)

    # ligtness method    
    new_image = (np.amax(input_image, axis=2) - np.amin(input_image, axis=2))/2

    # average method
    new_image = input_image/3

    # luminosity method
    new_image = 0.21*R + 0.72*G + 0.07*B

    # built-in opencv function 
    new_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    R, G, B = getColourChannels.getColourChannels(input_image)

    # Get the O1, O2 and O3 channels
    O_1 = (R - G) * 1/np.sqrt(2)
    O_2 = (R + G - 2 * B) * 1/np.sqrt(6)
    O_3 = (R + G + B) * 1/np.sqrt(3)

    new_image =  np.dstack((O_1, O_2, O_3))
    return new_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space
    R, G, B = getColourChannels.getColourChannels(input_image)

    # Get the normal RGB values
    R_ = R/(R + G + B)
    G_ = G/(R + G + B)
    B_ = B/(R + G + B)

    new_image =  np.dstack((R_, G_, B_))

    return new_image
