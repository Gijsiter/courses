import cv2
import numpy as np
from collections import defaultdict
from numpy import linalg as LA
import matplotlib.pyplot as plt


def get_keypoints(ims):
    keypoints = []
    detector = cv2.SIFT_create(contrastThreshold=0.06)
    for im in ims:
        kp, D = detector.detectAndCompute(im, None)
        keypoints.append((kp, D))
    return keypoints


def matches2pointpairs(matches, kp1, kp2):
    """
    Uses matches to construct point correspondences.
    """
    X1 = []
    X2 = []
    for match in matches:
        id1 = match.queryIdx
        id2 = match.trainIdx
        X1.append(kp1[id1].pt)
        X2.append(kp2[id2].pt)
    X1 = np.array(X1, dtype=np.float32)
    X2 = np.array(X2, dtype=np.float32)
    return X1, X2


def get_homographies(ims, pairs):
    ims_gray = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in ims])
    features = get_keypoints(ims_gray)
    homographies = {}
    matcher = cv2.BFMatcher(crossCheck=True)
    print("Estimating homographies.")
    for i, j in pairs:
        kp1, D1 = features[i]
        kp2, D2 = features[j]
        # Find matches
        matches = matcher.match(D1, D2)
        X1, X2 = matches2pointpairs(matches, kp1, kp2)
        H, inliers = cv2.findHomography(X1, X2, method=cv2.RANSAC)

        homographies[f'{i}-{j}'] = (H, inliers)

        if not np.any(np.all(pairs == [j, i], axis=1)):
            homographies[f'{j}-{i}'] = (LA.inv(H), inliers)
    print("Done.")
    return homographies, X1, X2

