import numpy as np
import cv2

# High value means no noise in signal so psnr has no importance
def myPSNR( orig_image, approx_image ):
    """
    orig_image: path to original image
    approx_image: path to approximated image
    """

    orig_image = cv2.imread(orig_image).astype(np.float32)
    approx_image = cv2.imread(approx_image).astype(np.float32)
    m,n, _ = orig_image.shape

    # MSE = np.sum((orig_image - approx_image)**2) / (m*n)
    MSE = np.mean((orig_image - approx_image)**2)

    PSNR = 20 * np.log10(orig_image.max() / np.sqrt(MSE))
    return PSNR
