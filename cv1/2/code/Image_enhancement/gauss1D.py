import numpy as np


def gauss1D(sigma , kernel_size):
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    # solution
    xmax  = int(np.floor(kernel_size/2))
    norm_term = 1 / (sigma*np.sqrt(2*np.pi))
    for i, x in enumerate(range(-xmax, xmax+1)):
        G[0, i] = norm_term*np.exp(-(x**2 / (2*sigma**2)))
    G = G / np.sum(G)

    return G
