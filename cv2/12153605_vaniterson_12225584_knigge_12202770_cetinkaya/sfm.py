import cv2
import argparse

from homography import *
from util import *
from fundamental import *
from chaining import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='Data/House/House', type=str)
    args = parser.parse_args()

    ims = ims2array(args.data_dir)

    # FUNDAMENTAL MATRIX ESTIMATION.
    pairs = np.array([[0,45]])
    H, X1, X2 = get_homographies(ims, pairs)
    F1 = estimate_fundamental(X1, X2, method='eight-point')
    F2 = estimate_fundamental(X1, X2, method='normalized-eight-point')
    F3, inliers = estimate_fundamental(X1, X2, method='nep-ransac')
    im1, im2 = ims[0], ims[45]

    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    i = 0
    j = -1

    # UNCOMMENT RULES BELOW TO SHOW PLOTS.
    print(F1, '\n')
    # plot_epipolar_lines(im1, im2, np.int32(X1)[i:j],
    #                     np.int32(X2)[i:j], F1)

    print(F2, '\n')
    # plot_epipolar_lines(im1, im2, np.int32(X1)[i:j],
    #                     np.int32(X2)[i:j], F2)

    print(F3, '\n\n')
    # plot_epipolar_lines(im1, im2, np.int32(X1)[i:j],
    #                     np.int32(X2)[i:j], F3)

    F4, inliers = cv2.findFundamentalMat(X1, X2, cv2.FM_8POINT)
    print(F4)
    # plot_epipolar_lines(im1, im2, np.int32(X1[i:j]),
    #                     np.int32(X2[i:j]), F4)


    # CHAINING
    PVM = point_view_mat(ims)
    PVM_bin = np.where(PVM != 0, 0, 1)
    # plt.imshow(PVM_bin, cmap='gray')
    # plt.tight_layout()
    # plt.show()

    P = factorize_and_stitch(PVM, setsize=4).T
    # draw_points(P)

    PVM1 = read_pvm()
    P = factorize_and_stitch(PVM1, setsize=PVM1.shape[0]).T
    # draw_points(P)
