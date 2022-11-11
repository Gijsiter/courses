import numpy as np
import random

from util import *
import matplotlib.pyplot as plt


def sampson_distance(X1, X2, F):
    """
    Computes the sampson distance.
    X1 and X2 are expected to be
    (matrices of) row vectors.
    """
    # Convert to homogeneous coordinates.
    X1, X2 = c2h(X1), c2h(X2)
    num = np.sum(X2 * (F@X1.T).T, axis=1)**2
    denom = np.sum(((F@X1.T)**2)[:2], axis=0) + np.sum(((F.T@X2.T)**2)[:2], axis=0)
    dists = num / denom
    return dists


def ransac(X1, X2, transform_fn, eval_fn, N=2000, P=8, threshold=3):
    most_inliers = 0
    inliers_best = None
    for i in range(N):
        samp_idx = random.sample(range(X1.shape[0]), P)
        S1, S2 = X1[samp_idx], X2[samp_idx]
        T = transform_fn(S1, S2)
        dists = eval_fn(X1, X2, T)
        inliers = np.where(dists < threshold, True, False)
        if sum(inliers) > most_inliers:
            inliers_best = inliers.copy()
            most_inliers = sum(inliers)

    T = transform_fn(X1[inliers_best], X2[inliers_best])

    return T, inliers_best