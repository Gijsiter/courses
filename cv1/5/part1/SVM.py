from collections import defaultdict
import numpy as np
from sklearn.svm import SVC, LinearSVC
from utils import random_split


def gather_splits(data, labels, npos, nneg):
    N = data.shape[0]
    frac_pos = (5*npos)/N
    (Xpos, Ypos), rem = random_split(data, labels, frac_train=frac_pos)
    Xneg, Yneg = np.split(rem[0], 5, axis=0), np.split(rem[1], 5)
    negs = []
    # Get chunks of negative examples.
    for c, cy in zip(Xneg, Yneg):
        M = len(c)
        chunks = int(M / nneg)
        negs.append([np.split(c, chunks, axis=0), np.split(cy, chunks)])
    splits = []
    final = []
    Xpos, Ypos = np.split(Xpos, 5, axis=0), np.split(Ypos, 5)
    # Create list of Pos/Neg splits.
    for i in range(4):
        xneg = None
        yneg = None
        for j in range(5):
            # Add chunk to negatives for 5th class.
            if i == j:
                final.append([negs[j][0][i], negs[j][1][i]])
            else:
                if xneg is not None:
                    xneg = np.vstack((xneg, negs[j][0][i]))
                    yneg = np.concatenate((yneg, negs[j][1][i]))
                else:
                    xneg, yneg = negs[j][0][i], negs[j][1][i]
        splits.append([(Xpos[i], Ypos[i]), (xneg, yneg)])
    # Add split for 5th class.
    xneg = np.vstack([X for (X, _) in final])
    yneg = np.concatenate([y for (_, y) in final])
    splits.append([(Xpos[-1], Ypos[-1]), (xneg, yneg)])

    return splits


def train_svms(data, labels, npos=50, nneg=50):
    """Train 5 binary SVMs"""
    splits = gather_splits(data, labels, npos, nneg)
    classifiers = defaultdict()
    data = []
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        classifiers[k] = defaultdict()
        for i, split in enumerate(splits):
            classifier = SVC(probability=True, kernel=k)
            X = np.vstack((split[0][0], split[1][0]))
            ypos = np.ones_like(split[0][1])
            yneg = np.zeros_like(split[1][1])
            y = np.concatenate((ypos, yneg))
            if k == 'linear':
                data.append([X, np.concatenate((split[0][1], split[1][1]))])
            classifier.fit(X, y)

            classifiers[k][i+1] = classifier

    return classifiers, data