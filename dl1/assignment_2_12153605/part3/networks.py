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
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch_geometric.nn as geom_nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_outputs: This number is required in order to specify the
                     output dimensions of the MLP
        TODO: 
        - define a simple MLP that operates on properly formatted QM9 data
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        self.net = []

        if not n_hidden:
            self.net.append(nn.Linear(n_inputs, n_outputs))
            self.net.append(nn.Softmax(dim=1))
        else:
            # Append first layer.
            self.net.append(nn.Linear(n_inputs, n_hidden[0]))
            self.net.append(nn.ReLU())
            # Append hidden layers.
            for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
                self.net.append(nn.Linear(n_in, n_out))
                self.net.append(nn.ReLU())
            # Append final layer.
            self.net.append(nn.Linear(n_hidden[-1], n_outputs))

        self.net = nn.Sequential(*self.net)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
            x: input to the network
        Returns:
            out: outputs of the network
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = self.net(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device


class GNN(nn.Module):
    """implements a graphical neural network in pytorch. In particular, we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden: int,
        n_output: int,
        num_convolution_blocks: int,
    ) -> None:
        """create the gnn

        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural networks (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. A block may include multiple convolutions
        
        TODO: 
        - define a GNN which has the following structure: node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - One the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        # Add embedder.
        self.net = [(nn.Embedding(n_node_features, n_hidden), 'x -> x')]
        # Add convolution blocks.
        for _ in range(num_convolution_blocks):
            conv_block = [
                nn.ReLU(),
                (geom_nn.RGCNConv(n_hidden, n_hidden, n_edge_features),
                 'x, edge_index, edge_attr -> x'),
                nn.ReLU(),
                (geom_nn.MFConv(n_hidden, n_hidden),
                 'x, edge_index -> x')
            ]
            self.net.extend(conv_block)
        # Add pooling.
        self.net.append((geom_nn.global_add_pool, 'x, batch_idx -> x'))
        # Add MLP.
        self.net.extend([
            (geom_nn.Linear(n_hidden, n_hidden), 'x -> x'),
            nn.ReLU(),
            (geom_nn.Linear(n_hidden, n_output), 'x -> x')
        ])
        self.net = geom_nn.Sequential('x, edge_index, edge_attr, batch_idx',
                                      self.net)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            prediction

        TODO: implement the forward pass being careful to apply MLPs only where they are allowed!

        Hint: remember to use global pooling.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = x.argmax(-1).long().view(-1)
        edge_attr = edge_attr.argmax(-1).long()
        out = self.net(x, edge_index, edge_attr, batch_idx)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
