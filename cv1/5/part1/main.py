from os import read
from numpy import random
from stl10_input import *
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import numpy as np
from collections import defaultdict
import pickle as pk
from utils import *
from SVM import *
from skimage.feature import hog


if __name__ == "__main__":
    #####################################
    #--- Load data and change labels ---#
    #####################################
    images = read_all_images(DATA_PATH)
    labels = read_labels(LABEL_PATH)
    relevant_classes = np.array([1, 2, 9, 7, 3])
    used_images, used_labels = keep_relevant_images(images, labels, relevant_classes)
    used_labels = assign_new_labels(used_labels, relevant_classes, [1,2,3,4,5])


    plot_five_classes(used_images, used_labels)


    #################################################################
    #--- Split data and perform K-means for a visual dictionary. ---#
    #################################################################
    # for k in [500, 1000, 2000]:
    #     model, split1dat, split2dat = BOW(used_images, used_labels, k=k,
    #                                     frac_train=0.5, frac_rem=1,
    #                                     descriptor_type='hog')
    #     pk.dump(model, open(f'hogkm{k}.pk', 'wb'))
    #     y2, imhists = split2dat[1], [pair[1] for pair in split2dat[-1]]
    #     pk.dump((split2dat[0], imhists, y2), open(f'hogtrdat{k}.pk', 'wb'))


    ######################################################
    #--- Load existing training data for classifiers. ---#
    ######################################################
    kmeans = pk.load(open('HOG/kmeans/hogkm500.pk', 'rb'))
    train_images, train_hists, train_labels = pk.load(
                                            open('HOG/data/hogtrdat500.pk', 'rb')
                                            )
    plot_histograms(np.array(train_hists), train_labels)

    ############################
    #--- Train classifiers. ---#
    ############################
    # classifiers, svmdata = train_svms(np.array(train_hists), train_labels)
    # pk.dump([classifiers, svmdata], open('svms.pk', 'wb'))


    ###############################
    #--- Load pretrained SVMs. ---#
    ###############################
    classifiers, svmdata = pk.load(open('HOG/svms/hogsvms500.pk', 'rb'))


    ##########################################################################
    #--- Create histograms of test data or load existing test histograms. ---#
    ##########################################################################
    test_images = read_all_images('data/stl10_binary/test_X.bin')
    test_labels = read_labels('data/stl10_binary/test_y.bin')
    test_images, test_labels = keep_relevant_images(test_images, test_labels,
                                                    relevant_classes)
    test_labels = assign_new_labels(test_labels, relevant_classes, [1,2,3,4,5])
    # test_hists = np.array([im2hist(im, kmeans, 'hog') for im in test_images])
    # pk.dump((test_hists, test_labels), open('hogntestdat2000.pk', 'wb'))


    ########################################################
    #--- Load existing test data histograms and labels. ---#
    ########################################################
    test_hists, test_labels = pk.load(open('HOG/data/hogtestdat500.pk', 'rb'))


    ###############################################################################
    #--- Classify data, create rankings and compute the Mean Average Precision ---#
    ###############################################################################
    probs_test, probs_train = predict_probs(classifiers, test_hists, svmdata)
    rankings_train, rankings_test = compute_rankings(probs_train, probs_test,
                                                     svmdata, test_labels)
    MAPtrain, MAPtest = computeMAPs(rankings_train, rankings_test)

    for key in MAPtest:
        print('Set: train', '\nMAPs:', np.array(MAPtrain[key]),
              '\nKernel type:', key, '\nMean MAP:', np.mean(MAPtrain[key]), '\n\n')
        print('Set: test', '\nMAPs:', np.array(MAPtest[key]),
              '\nKernel type:', key, '\nMean MAP:', np.mean(MAPtest[key]), '\n\n')

    for i in range(5):
        top = test_images[rankings_test['rbf'][i][0][:5]]
        bottom = test_images[rankings_test['rbf'][i][0][:-6:-1]]
        plot_top_bottom_5(top, bottom, i+1, 'HOG')