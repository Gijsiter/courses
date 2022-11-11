import numpy as np


def gauss1D(sigma ,kernel_size):
    G = np.zeros((1, kernel_size))
    if kernel_size % 2 == 0:
        raise ValueError("""kernel_size must be odd, otherwise the filter \
                         will not have a center to convolve on""")

    xmax  = int(np.floor(kernel_size/2))
    norm_term = 1 / (sigma*np.sqrt(2*np.pi))
    for i, x in enumerate(range(-xmax, xmax+1)):
        G[0, i] = norm_term*np.exp(-(x**2 / (2*sigma**2)))
    G = G / np.sum(G)

    return G

def  gauss2D(sigma ,kernel_size):
    sigma_x, sigma_y = sigma
    Gx = gauss1D(sigma_x, kernel_size)
    Gy = gauss1D(sigma_y, kernel_size)
    G = Gx.T@Gy

    return G

def createGabor(lamb, theta, psi, sigma, gamma, kernel_shape):
    if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0:
        raise ValueError("""kernel_dimensions must be odd, otherwise the filter \
                         will not have a center to convolve on""")
    Gabor = np.zeros(kernel_shape[::-1])
    xmax = int(np.floor(kernel_shape[0] / 2))
    ymax = int(np.floor(kernel_shape[1] / 2))
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, sin], [-sin, cos]])
    for i, y in enumerate(np.arange(ymax, -ymax-1, -1)):
        for j, x in enumerate(np.arange(-xmax, xmax+1)):
            x_prime, y_prime = rot@[x, y]
            gauss = np.exp(-((x_prime**2 + gamma**2 * y_prime**2) / 2*sigma**2))
            Gabor[i,j] = gauss*np.cos(2*np.pi*(x_prime/lamb) + psi) \
                         + gauss*np.sin(2*np.pi*(x_prime/lamb) + psi)

    return Gabor