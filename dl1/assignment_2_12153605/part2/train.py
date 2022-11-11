###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Adapted: 2021-11-11
###############################################################################

from datetime import datetime
import argparse

from torch.utils import data
from torch._C import dtype
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(dataset, args.batch_size, 
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn, num_workers=3)
    # Create model
    args.vocabulary_size = dataset.vocabulary_size
    model = TextGenerationModel(args)
    model = model.to(args.device)
    model.train()
    # Create optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    # Training loop
    train_losses = []
    accuracies = []
    generated_seqs = []
    # Perform training procedure
    for epoch in range(args.num_epochs):
        epoch_losses = []
        # # Fit training data set.
        for X, T in data_loader:
            X = X.to(args.device)
            T = T.to(args.device)
            out = model(X)

            loss = criterion(torch.log(out.view(-1, out.shape[-1])), T.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        # Compute accuracy and generate samples.
        model.eval()
        with torch.no_grad():
            accuracy = evaluate_model(model, data_loader, args.device)
            accuracies.append(accuracy)
            if (epoch + 1) in [1, 7, 20]:
                seqs = model.sample(sample_length=args.input_seq_length, batch_size=5).detach().cpu().numpy()
                gen_seqs = [dataset.convert_to_string(seqs[:, i].squeeze())
                            for i in range(seqs.shape[1])]
                generated_seqs.append(gen_seqs)
        model.train()
    state = {'model': model.state_dict(), 'train_losses': train_losses,
             'accuracies': accuracies, 'generated_seqs': generated_seqs}
    torch.save(state, f"{args.checkpoint_name}.pt")
    #######################
    # END OF YOUR CODE    #
    #######################


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    total_acc = 0
    with torch.no_grad():
        for X, t in data_loader:
            X = X.to(device)
            t = t.to(device)
            predictions = model(X)
            char_predictions = torch.argmax(predictions, dim=-1)
            total_acc += torch.mean((char_predictions == t).type(torch.float64)).item()

        accuracy = total_acc / len(data_loader)
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    parser.add_argument('--checkpoint_name', type=str,
                        help='Filename for saving the model')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    train(args)
