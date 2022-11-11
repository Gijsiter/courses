import cv2
import matplotlib.pyplot as plt
import matplotlib


def denoise( image, kernel_type, **kwargs):
    image = cv2.imread(image)
    if kernel_type == 'box':
        imOu = cv2.blur(image, kwargs['kernel_size'])
    elif kernel_type == 'median':
        imOu = cv2.medianBlur(image, kwargs['kernel_size'])
    elif kernel_type == 'gaussian':
        imOu = cv2.GaussianBlur(image,kwargs['kernel_size'],kwargs['sigma'])
    else:
        print('Operation not implemented')
    return imOu
