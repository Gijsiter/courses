import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import open3d as o3d


def ims2array(data_dir):
    """
    Returns an array of N x (H x W x C),
    where array[i] is the ith image
    in data_dir.
    """
    for (_, _, filenames) in os.walk(data_dir):
        names = sorted(filenames)
        break
    ims = []
    for name in names:
        ims.append(cv2.imread(f'{data_dir}/{name}'))
    ims = np.stack(ims)
    return ims


def get_transformation_pairs(n_frames):
    frames = np.arange(n_frames).reshape(-1, 1)
    pairs = np.hstack([np.repeat(frames, n_frames, axis=0),
                       np.tile(frames, (n_frames, 1))])
    redundant = np.where(pairs[:,0] >= pairs[:,1], True, False)
    return pairs[~redundant]


def c2h(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack([X, ones])


def h2c(X):
    return (X[:,:2]/X[:,2:3])


def read_pvm():
    PVM = []
    with open('PointViewMatrix.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            PVM.append(np.array(line.split(), dtype=np.float32))
    f.close()
    PVM = np.array(PVM)
    return PVM


#**********************************************************************#
# The code for the functions drawlines and 
# plot_epipolar_lines have been obtained from:
# https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
#**********************************************************************#

def drawlines(img1, img2, lines, X1, X2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, X1, X2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def plot_epipolar_lines(img1, img2, X1, X2, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(X2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    print(lines1.shape)
    img5, _ = drawlines(img1, img2, lines1, X1, X2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(X1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img3, _ = drawlines(img2, img1, lines2, X2, X1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


def draw_points(A):
    A[2] = A[2]
    vis = o3d.geometry.PointCloud()
    vis.points = o3d.utility.Vector3dVector(A.T)

    o3d.visualization.draw_geometries([vis])

    return None