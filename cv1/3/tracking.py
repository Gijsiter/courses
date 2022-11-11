from PIL import Image
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob
from scipy.signal import convolve2d
import matplotlib.cm as cm
import matplotlib.animation as animation

from harris_corner_detector import HCD


def divide_regions(img, points=None, window_size=15):
    """
    img: image matrix
    points: tuple of vectors containing row and column coordinates.
    """
    regions = []
    if points is not None:
        rows, cols = points
        pad = window_size // 2
        im_padded = np.pad(img, ((pad, pad), (pad, pad)))
        for r, c in zip(rows, cols):
            regions.append(im_padded[r:r+16, c:c+16])
    else:
        for i in range(0, img.shape[0], window_size):
            for j in range(0, img.shape[1], window_size):
                regions.append(img[i:i+window_size, j:j+window_size])

    return regions


def compute_Ab(img1, img2):
    I_x = scipy.signal.convolve2d(img1, np.array([[-1,0,1]]), mode='same')
    I_y = scipy.signal.convolve2d(img1, np.array([[-1],[0],[1]]), mode='same')
    A = np.dstack((I_x.flatten(), I_y.flatten()))[0]

    I_t = img1 - img2
    
    b = -1*I_t.flatten()

    return np.linalg.inv(A.T@A)@A.T@b

# Import the pictures
# glob_list = glob.glob('images/toy/*.jpg')
glob_list = glob.glob('images/doll/*.jpg')


# Iterate over the pictures and use the algorithms
imgs = []
new_img = []
fig = plt.figure()
r = c = None
u = v = None
for i in range(len(glob_list) - 1):
    img = cv2.cvtColor(cv2.imread(glob_list[i]), cv2.COLOR_BGR2RGB)
    
    # Use the harris corner detector
    if r is None:
        H, r, c = HCD(img, threshold=10000, n=3, visualize=False)
        r, c = np.array(r), np.array(c)
    else:
        r = (r + v).astype(np.int)
        c = (c + u).astype(np.int)


    # Give image path to the lucas kanade algorithm
    img1_path = glob_list[i]
    img2_path = glob_list[i+1]

    # make 2d array from images and float 32
    img1 = cv2.cvtColor(cv2.imread(img1_path),
                        cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    img2 = cv2.cvtColor(cv2.imread(img2_path),
                        cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    regions1 = divide_regions(img1, (r, c))
    regions2 = divide_regions(img2, (r, c))
    # Calculate optical flow for each region
    of = []
    for region1, region2 in zip(regions1, regions2):
        of.append(compute_Ab(region1, region2))

    # plot image
    z = 0
    X = r
    Y = c
    X_ = []
    Y_ = []
    for y in Y:
        X_.append(X)
        Y_.append(len(X) * [y])
    X_ = np.array(X_).flatten()
    Y_ = np.array(Y_).flatten()
    u = [i[0] for i in of]
    v = [i[1] for i in of]
    # Create the frames of the video
    new_img.append([plt.imshow(img, cmap=cm.Greys_r, animated=True),
                    plt.scatter(c, r, c='r', s=5, marker='o'),
                    plt.quiver(c, r, [i[0] for i in of], [i[1] for i in of])])

# Create the animation
ani = animation.ArtistAnimation(fig, new_img, interval=200, blit=True, repeat_delay=1000)
ani.save('doll.mp4')
plt.show()