import numpy as np
from numpy import linalg as LA
import cv2
from scipy.spatial import procrustes

from homography import get_keypoints


def update_PVM(PVM, matches, kp, kp2col):
    # Indices of matches keypoints in second view.
    matched = [match.trainIdx for match in matches]
    all_points = np.arange(len(kp))
    new_points = all_points[~np.isin(all_points, matched)]

    # Create space for new keypoints and add values to existing columns.
    N_old = PVM.shape[1]
    PVM = np.pad(PVM, [(0,0), (0, len(new_points))])
    N_new = PVM.shape[1]
    new_entry = np.zeros((2, N_new))
    kp2col_new = {}
    # Add matches to existing columns
    for match in matches:
        kp_1_idx = match.queryIdx
        kp_idx = match.trainIdx
        pvm_idx = kp2col[kp_1_idx]
        new_entry[:, pvm_idx] = kp[kp_idx].pt
        kp2col_new[kp_idx] = pvm_idx
    # Add new points to new columns
    for kp_idx, c in zip(new_points, range(N_old, N_new)):
        new_entry[:, c] = kp[kp_idx].pt
        kp2col_new[kp_idx] = c

    PVM = np.vstack([PVM, new_entry])

    return PVM, kp2col_new


def point_view_mat(ims):
    ims_gray = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in ims])
    features = get_keypoints(ims_gray)
    idx = np.arange(len(features))
    matcher = cv2.BFMatcher(crossCheck=False)

    # Create initial 2 rows.
    kp_1, D_1 = features[0]
    kp, D = features[1]
    matches = matcher.match(D_1, D)

    PVM = np.hstack([kpt.pt for kpt in kp_1]).reshape((2,-1), order='F')
    kp2col = {c : c for c in range(len(kp_1))}
    PVM, kp2col = update_PVM(PVM, matches, kp, kp2col)
    D_1 = features[1][1]
    # Create rest of the matrix.
    for j in idx[2:]:
        kp, D = features[j]
        matches = matcher.match(D_1, D)
        PVM, kp2col = update_PVM(PVM, matches, kp, kp2col)
        D_1 = D
        kp_1 = kp
    # Compare final and first view.
    kp, D = features[0]
    matches = matcher.match(D_1, D)
    for match in matches:
        kp_1_idx = match.queryIdx
        kp_idx = match.trainIdx
        pvm_idx = kp2col[kp_1_idx]
        PVM[:2, pvm_idx] = kp[kp_idx].pt

    return PVM


def factorize(M):
    M = M - np.mean(M, axis=1, keepdims=True)
    U, W, Vt = LA.svd(M)
    U3 = U[:,:3]
    W3 = np.diag(W[:3])
    Vt3 = Vt[:3]

    M = U3 @ np.sqrt(W3)
    S = np.sqrt(W3) @ Vt3

    return M, S


def factorize_and_stitch(PVM, setsize=3):
    # Split PVM to setsize-view subblocks.
    ids = np.arange(PVM.shape[0])[2*setsize::2*setsize]
    sets = np.split(PVM, ids)
    SS = []
    range = np.arange(PVM.shape[1])
    # Factorize each block and keep points.
    for s in sets:
        if s.shape[0] == 2:
            continue
        mask = np.all(s != 0, axis=0)
        s = s[:, mask]
        _, S = factorize(s)
        SS.append((S.T, range[mask]))
    T, ids = SS[0]
    points = T
    # Map points to first cloud.
    for S, ids1 in SS:
        P = T[np.isin(ids, ids1)]
        P1 = S[np.isin(ids1, ids)]
        _, m2, _ = procrustes(P, P1)
        points = np.vstack([points, m2])

    return points