################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch
import pickle as pk
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Run all hyperparameter configurations as requested
    hidden_setups = [[128], [256, 128], [512, 256, 128]]
    results = []
    for hidden_dims in hidden_setups:
        _, _, _, log_bn = train_mlp_pytorch.train(hidden_dims, 0.1, True, 128,
                                                  20, 42, 'data/')
        _, _, _, log_reg = train_mlp_pytorch.train(hidden_dims, 0.1, False, 128,
                                                   20, 42, 'data/')
        results.append((hidden_dims, {'bn': log_bn, 'reg': log_reg}))
    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    pk.dump(results, open(results_filename, 'wb'))
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    results = pk.load(open(results_filename, 'rb'))
    coords = [[(0,0), (0,1)],
              [(1,0), (1,1)],
              [(2,0), (2,1)]]
    ran = np.arange(20)
    fig, ax = plt.subplots(3,2, sharex=True)
    for cpair, res in zip(coords, results):
        ax[cpair[0]].plot(ran, res[1]['reg']['train_accs'],
                          label='W/o BN')
        ax[cpair[1]].plot(ran, res[1]['reg']['val_accs'],
                          label='W/o BN')
        ax[cpair[0]].set_title("Training accuracy, hidden units: {}".format(res[0]))
        ax[cpair[0]].set_ylabel("Accuracy")

        ax[cpair[0]].plot(ran, res[1]['bn']['train_accs'],
                          label='With BN')
        ax[cpair[1]].plot(ran, res[1]['bn']['val_accs'],
                          label='With BN')
        ax[cpair[1]].set_title("Validation accuracy, hidden units: {}".format(res[0]))

        ax[cpair[0]].legend()
        ax[cpair[1]].legend()

    ax[coords[-1][0]].set_xlabel("Epoch")
    ax[coords[-1][1]].set_xlabel("Epoch")
    plt.xticks(ran, ran + 1)
    plt.show()
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'bn_results.pk'
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)