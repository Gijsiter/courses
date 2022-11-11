import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import KDTree


def euclidean_matching(source, target, R, t):
    P = R @ source + t
    matches = [np.argmin(
        LA.norm(target.T - s, axis=1)) for s in P.T
    ]
    tgt_sorted = target[:, matches]
    return source, tgt_sorted, P


def kdtree_matching(source, target, R, t):
    P = R @ source + t
    tree = KDTree(target.T)
    _, matches = tree.query(P.T, k=1)
    return source, target[:, matches.flatten()], P


def get_cells(union, H, W):
    minx, maxx = np.amin(union[0]), np.amax(union[0])
    miny, maxy = np.amin(union[1]), np.amax(union[1])
    # Create HxW cells
    cellsH, dH = np.linspace(miny, maxy, num=H, endpoint=False, retstep=True)
    cellsW, dW = np.linspace(minx, maxx, num=W, endpoint=False, retstep=True)
    XX, YY = np.meshgrid(cellsW, cellsH)
    cells = np.vstack([XX.flatten(), YY.flatten()]).T
    return cells, dH, dW


def closest_to_cell(X_or, X, cell, dW, dH):
    # Get projections falling into cell
    xcondition = np.logical_and(cell[0] <= X[0], X[0] < cell[0] + dW)
    candidates = X[:, xcondition]
    ycondition = np.logical_and(
        cell[1] <= candidates[1], candidates[1] < cell[1] + dH
    )
    candidates = candidates[:, ycondition]
    # Return arrays with inf if no point falls into cell
    if candidates.shape[1] == 0:
        infarray = np.full((1, X.shape[0]), np.inf)
        if X_or is not None:
            return infarray, infarray
        return infarray
    # Else return point closest w.r.t. projection axis (Z).
    closest = candidates[:, np.argmin(abs(candidates[-1]))]

    if X_or is not None:
        closest_or = X_or.T[np.where(np.all(X.T == closest, axis=1))]
        return closest, closest_or

    return closest


def get_window(w, h, H, W):
    WW, HH = np.meshgrid(np.arange(w-3, w+4), np.arange(h-3, h+4))
    cells = np.vstack([WW.flatten(), HH.flatten()])

    Winvalid = np.logical_or(cells[0] < 0, cells[0] >= W)
    Hinvalid = np.logical_or(cells[1] < 0, cells[1] >= H)
    valid = ~np.logical_or(Winvalid, Hinvalid)

    cells = cells[:, valid]
    return cells


def zbuffer_matching(source, target, R, t, H=20, W=20):
    D = source.shape[0]
    # Take union and define minimal bounding box.
    P = R @ source + t
    union = np.hstack([P, target])
    cells, dH, dW = get_cells(union, H, W)

    # Project points to cells.
    src_proj = [closest_to_cell(source, P, cell, dW, dH) for cell in cells]
    source = np.vstack([p[1] for p in src_proj]).reshape(W, H, -1)
    src_buffer = np.vstack([p[0] for p in src_proj]).reshape(W, H, -1)

    tgt_buffer = np.vstack([
        closest_to_cell(None, target, cell, dW, dH) for cell in cells]
    ).reshape(W, H, -1)

    # For each cell, find closest point (iff the cell contains candidates).
    tgt_sorted = []
    remove = ([], [])
    for w in range(W):
        for h in range(H):
            ws, hs = get_window(w, h, H, W)
            vecs = tgt_buffer[ws, hs]
            vecs = vecs[~np.any(vecs == np.inf, axis=1)]

            if vecs.size == 0 or np.any(src_buffer[w, h] == np.inf):
                remove[0].append(w)
                remove[1].append(h)
                continue
            closest = vecs[np.argmin(LA.norm(vecs - src_buffer[w, h], axis=1))]
            tgt_sorted.append(closest)

    tgt_sorted = np.array(tgt_sorted)
    # Remove source points that could not be matched.
    rm_mask = np.ones_like(src_buffer)
    rm_mask[remove] = False
    rm_mask = rm_mask.reshape(-1, D)

    src_buffer = src_buffer.reshape(-1, D)[np.all(rm_mask, axis=1)]
    source = source.reshape(-1, D)[np.all(rm_mask, axis=1)]

    return source.T, tgt_sorted.T, src_buffer.T