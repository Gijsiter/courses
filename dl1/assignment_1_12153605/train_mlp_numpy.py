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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils
import pickle as pk
import time
import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    class_prediction = np.argmax(predictions, axis=1)
    accuracy = np.mean(class_prediction == targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    Nb = len(data_loader)
    avg_accuracy = 0
    for X, t in data_loader:
        out = model.forward(X.reshape(X.shape[0], -1))
        avg_accuracy += (1 / Nb) * accuracy(out, t)
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir, save_to=None):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
      log_dir: Directory where to store the output.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Initialize model and loss module
    model = MLP(3072, hidden_dims, 10)
    loss_module = CrossEntropyModule()
    # Training loop including validation
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val = None
    best_model = None
    # Perform training procedure
    for i in range(epochs):
        total_loss = 0
        # Fit training data set.
        for X, t in cifar10_loader['train']:
            out = model.forward(X.reshape(X.shape[0], -1))

            loss = loss_module.forward(out, t)
            total_loss += loss

            lossgrad = loss_module.backward(out, t)
            model.backward(lossgrad)
            for module in model.modules:
                if type(module) == LinearModule:
                    module.params['weight'] = module.params['weight'] \
                                              - lr*module.grads['weight']
                    module.params['bias'] = module.params['bias'] \
                                            - lr*module.grads['bias']
            model.clear_cache()
        # Evaluate on validation set.
        val_acc = evaluate_model(model, cifar10_loader['validation'])
        if best_val is None or val_acc > best_val:
            best_val = val_acc
            best_model = deepcopy(model)
        # Update logging info
        train_accuracies.append(evaluate_model(model, cifar10_loader['train']))
        train_losses.append(total_loss)
        val_accuracies.append(val_acc)
    # Test best model
    model = best_model
    test_accuracy = evaluate_model(model, cifar10_loader['test'])
    # Add any information you might want to save for plotting
    logging_dict = {'train_accs': train_accuracies,
                    'train_losses': train_losses,
                    'val_accs': val_accuracies,
                    'test_accuracy': test_accuracy,}
    if save_to is not None:
        state = {'model': model, 'log': logging_dict}
        pk.dump(state, open(f"{save_to}.pk", 'wb'))
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')
    parser.add_argument('--save_to', default=None, type=str,
                        help="Directory to save the log.")

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    