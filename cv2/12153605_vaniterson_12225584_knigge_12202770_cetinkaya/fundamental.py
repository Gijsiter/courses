import numpy as np
from numpy import linalg as LA
from functools import partial

from util import c2h
from ransac import *
from homography import get_keypoints


def construct_A(X1, X2):
    col1 = X1[:, [0]] * X2[:, [0]]
    col2 = X1[:, [0]] * X2[:, [1]]
    col3 = X1[:, [0]]
    col4 = X1[:, [1]] * X2[:, [0]]
    col5 = X1[:, [1]] * X2[:, [1]]
    col6 = X1[:, [1]]
    col7 = X2[:, [0]]
    col8 = X2[:, [1]]
    col9 = np.ones_like(col1)

    A = np.hstack([
        col1, col2, col3, col4, col5, col6, col7, col8, col9
    ])
    return A


def construct_T(X):
    m = np.mean(X, axis=0)
    d = np.mean(np.sqrt(np.sum((X - m)**2, axis=1)))
    mx, my = m
    s2d = np.sqrt(2) / d
    T = np.array([[s2d, 0, -mx*s2d],
                  [0, s2d, -my*s2d],
                  [0, 0, 1]
    ])
    return T


def eight_point(X1, X2):
    A = construct_A(X1, X2)
    _, _, Vat = LA.svd(A)
    F = Vat[-1].reshape((3, 3), order='F')
    Uf, Df, Vft = LA.svd(F)
    Df[-1] = 0.0
    F = Uf @ np.diag(Df) @ Vft
    return F


def normalized_eight_point(X1, X2, T1=None, T2=None):
    if T1 is None and T2 is None:
        T1 = construct_T(X1)
        T2 = construct_T(X2)
        X1_hat = c2h(X1) @ T1.T
        X2_hat = c2h(X2) @ T2.T
    else:
        X1_hat = c2h(X1) @ T1.T
        X2_hat = c2h(X2) @ T2.T
    F_hat = eight_point(X1_hat, X2_hat)
    F = T2.T @ F_hat @ T1
    # # print verifying the mean and average distance.
    # X1mean = np.mean(h2c(X1_hat), axis=0)
    # print()
    # print("p_hat mean x,y: ", np.mean(h2c(X1_hat), axis=0))
    # avgdist = np.mean(np.sqrt(np.sum((h2c(X1_hat) - X1mean)**2,axis=1)))
    # print("p_hat avg dist: ", avgdist)
    # print()
    return F


def NEP_ransac(X1, X2):
    T1 = construct_T(X1)
    T2 = construct_T(X2)
    transform_fn = partial(normalized_eight_point, T1 = T1, T2 = T2)
    F, inliers = ransac(X1, X2, transform_fn, sampson_distance)
    return F, inliers


def estimate_fundamental(X1, X2, method='eight-point'):
    if method == 'eight-point':
        F = eight_point(X1, X2)
    elif method == 'normalized-eight-point':
        F = normalized_eight_point(X1, X2)
    elif method == "nep-ransac":
        F, inliers = NEP_ransac(X1, X2)
        return F, inliers
    else:
        raise NotImplementedError("Unknown estimation method")
    return F