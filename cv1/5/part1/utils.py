import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.feature import hog


def extract_features(img, type='sift'):
    if type == 'sift':
        # Convert to greyscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Create feature extractor
        extractor = cv2.SIFT_create()
        keypoints, descriptors = extractor.detectAndCompute(gray_img, None)

        return keypoints, descriptors
    elif type == 'hog':
        hogs = hog(img, orientations=20, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), feature_vector=False)
        hogs = hogs.reshape(-1, 20)
        return 0, hogs
    else:
        print("Invalid descriptor type")


def plot_features(img, keypoints, plot=True):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgC = cv2.UMat(img)
    # cv2.imshow('a', img)

    if plot:
        plt.imshow(cv2.drawKeypoints(gray, keypoints, imgC))
        plt.show()
    else:
        return cv2.drawKeypoints(imgC, keypoints, imgC)


def get_class_indices(used_labels):
    # Get label-indices for each class.
    AP = np.where(used_labels == 1)[0]
    BR = np.where(used_labels == 2)[0]
    SH = np.where(used_labels == 3)[0]
    HS = np.where(used_labels == 4)[0]
    CR = np.where(used_labels == 5)[0]

    return AP, BR, SH, HS, CR


def assign_new_labels(used_labels, old, new):
    new_labels = np.zeros_like(used_labels)
    for i, j in zip(old, new):
        new_labels[used_labels == i] = j

    return new_labels


def plot_five_classes(used_images, used_labels):
    # Find images
    AP, BR, SH, HS, CR = get_class_indices(used_labels)

    fig, axs = plt.subplots(2,5, sharex=True, sharey=True)

    res = []
    for index in [AP[0], AP[1], BR[0], BR[1], SH[0], SH[1], 
                  HS[0], HS[1], CR[0], CR[1]]:
        keypoints, descriptors = extract_features(used_images[index])
        res.append(plot_features(used_images[index], keypoints, plot=False).get())

    axs[0,0].imshow(res[0])
    axs[1,0].imshow(res[1])
    axs[0,0].set_title("Airplane")

    axs[0,1].imshow(res[2])
    axs[1,1].imshow(res[3])
    axs[0,1].set_title("Bird")

    axs[0,2].imshow(res[4])
    axs[1,2].imshow(res[5])
    axs[0,2].set_title("Ship")

    axs[0,3].imshow(res[6])
    axs[1,3].imshow(res[7])
    axs[0,3].set_title("Horse")

    axs[0,4].imshow(res[8])
    axs[1,4].imshow(res[9])
    axs[0,4].set_title("Car")

    plt.show()

    return None


def random_split(images, labels, seed=0, frac_train=0.5):
    # Split data into 2 parts with equal class representation.
    np.random.seed(seed)
    split1, split2 = [], []
    # Randomly split images.
    C_indices = get_class_indices(labels)
    N = C_indices[0].shape[0]
    randrange = np.random.permutation(np.arange(N))
    len1 = int(N*frac_train)
    split1_idx, split2_idx = np.array_split(randrange, [len1])

    # Create a split for each class
    for indices in C_indices:
        split1.append((images[indices[split1_idx]],
                       labels[indices[split1_idx]]))
        split2.append((images[indices[split2_idx]],
                       labels[indices[split2_idx]]))

    # Gather images and labels into 2 arrays per split.
    split1 = (np.vstack([pair[0] for pair in split1]),
              np.concatenate([pair[1] for pair in split1]))
    split2 = (np.vstack([pair[0] for pair in split2]),
              np.concatenate([pair[1] for pair in split2]))

    return split1, split2


def BOW(images, labels , k=500, frac_train=0.5, frac_rem=1, descriptor_type='sift'):
    (X1, y1), (X2, y2) = random_split(images, labels, frac_train=frac_train)
    descriptors = None
    for image in X1:
        _, desc = extract_features(image, type=descriptor_type)
        if descriptors is None:
            descriptors = desc
        else:
            descriptors = np.vstack([descriptors, desc])

    kmeans = KMeans(n_clusters=k, random_state=0).fit(descriptors)

    # Create visual vocabularies and their histograms for remaining images.
    M = int(X2.shape[0]*frac_rem)
    BoWs = []
    for image in X2[:M]:
        _, desc = extract_features(image, type=descriptor_type)
        labs = kmeans.predict(desc)
        visvoc = defaultdict(lambda: [])
        keys = np.unique(labs)
        for key in keys:
            visvoc[key] = desc[labs == key]
        hist = np.bincount(labs, None, k)
        hist = hist / sum(hist)
        BoWs.append((visvoc, np.array(hist)))

    return kmeans, (X1, y1), (X2[:M], y2[:M], BoWs)


def im2hist(image, kmeans_model, descriptor_type='sift'):
    _, desc = extract_features(image, type=descriptor_type)
    labs = kmeans_model.predict(desc)
    hist = np.bincount(labs, None, kmeans_model.n_clusters)
    hist = hist / sum(hist)

    return hist


def plot_histograms(freq_vecs, labels):
    """
    :param freq_vecs: list of frequency vectors
    :param labels:    respective labels for frequency vectors
    """
    labs = np.unique(labels)
    bins = np.arange(len(freq_vecs[0]))
    _, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    cx, cy = np.meshgrid(range(3), range(2))
    # Create normalized histograms for each class.
    for coord, l in zip(zip(cy.flatten(), cx.flatten()), labs):
        x = sum(freq_vecs[labels == l])

        ax[coord].bar(bins, x)
        ax[coord].set_title(f"Class {l}")
        ax[coord].set_xlabel("Visual words")

    ax[0,0].set_ylabel("Frequency")
    ax[1,0].set_ylabel("Frequency")
    ax[-1, -1].axis('off')
    plt.show()

    return None


def mAP(classlabel, labels, M):
    f = 0
    map = 0
    for i, l in enumerate(labels):
        if l == classlabel:
            f += 1
            map += (f / (i+1))
        else:
            map += 0

    return map / M


def plot_top_bottom_5(top_ims, bottom_ims, label, descriptor_type):
    classes = {1: 'airplanes', 2: 'birds', 3: 'ships', 4: 'horses', 5: 'cars'}
    fig, axs = plt.subplots(2,5, sharex=True, sharey=True)
    fig.suptitle(f"Top and bottom 5 ranked with classifier for {classes[label]} "
                 f"using {descriptor_type} descriptors.")

    axs[0,0].set_ylabel("Top 5")
    axs[1,0].set_ylabel("Bottom 5")

    axs[0,0].imshow(top_ims[0])
    axs[1,0].imshow(bottom_ims[0])

    axs[0,1].imshow(top_ims[1])
    axs[1,1].imshow(bottom_ims[1])

    axs[0,2].imshow(top_ims[2])
    axs[1,2].imshow(bottom_ims[2])

    axs[0,3].imshow(top_ims[3])
    axs[1,3].imshow(bottom_ims[3])

    axs[0,4].imshow(top_ims[4])
    axs[1,4].imshow(bottom_ims[4])

    plt.show()

    return None


def predict_probs(classifiers, test_hists, svmdata):
    probs_test = {
        k : [classifiers[k][i].predict_proba(test_hists)[:,1] for i in range(1,6)]
        for k in classifiers.keys()
    }
    probs_train = {
        k : [classifiers[k][i+1].predict_proba(svmdat[0])[:,1]
        for i, svmdat in enumerate(svmdata)]
        for k in classifiers.keys()
    }

    return probs_test, probs_train

def compute_rankings(probs_train, probs_test, svmdata, test_labels):
    rankings_test = {}
    rankings_train = {}

    for i, (k, k1) in enumerate(zip(probs_train, probs_test)):
        pred_tr = probs_train[k]
        pred_test = probs_test[k1]
        ranks_train = []
        ranks_test = []
        for i, (ptest, ptrain) in enumerate(zip(pred_test, pred_tr)):
            sort_idx_test = np.argsort(ptest)[::-1]
            sort_idx_tr = np.argsort(ptrain)[::-1]
            ranks_train.append(
                [svmdata[i][0][sort_idx_tr], svmdata[i][1][sort_idx_tr]]
            )
            ranks_test.append(
                [sort_idx_test, test_labels[sort_idx_test]]
            )
        rankings_train[k] = ranks_train
        rankings_test[k1] = ranks_test

    return rankings_train, rankings_test


def computeMAPs(rankings_train, rankings_test):
    # Compute MAP for each classifier.
    MAPtrain = {}
    MAPtest = {}
    np.set_printoptions(precision=3)
    for k, k1 in zip(rankings_train, rankings_test):
        ranks_train = rankings_train[k]
        ranks_test = rankings_test[k1]
        mapstr = [mAP(i+1, rank[1], 50) for i, rank in enumerate(ranks_train)]
        maptst = [mAP(i+1, rank[1], 800) for i, rank in enumerate(ranks_test)]
        MAPtrain[k] = mapstr
        MAPtest[k1] = maptst

    return MAPtrain, MAPtest