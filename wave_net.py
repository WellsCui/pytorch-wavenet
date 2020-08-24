#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causal_conv1d import CausalConv1d
from wave_net_layer import WaveNetLayer
from utils import pad_source
from typing import List, Tuple, Dict, Set, Union, Optional
from multiprocessing import Process, Queue

class WaveNet(nn.Module):
    """ Simple CNN Model:
        - 1D CNN
    """

    def __init__(self, 
        layers=30, 
        block_size=10,
        kernel_size=3,
        residual_channels=256,
        layer_channels=256, 
        skip_channels=256,
        aggregate_channels=256,
        classes=40,
        ):
        """ Init CNN Model.

        @param layers (int): Number of layers
        @param context_size (int): Size of conditional context
        """
        super(WaveNet, self).__init__()

        self.layers = layers
        self.block_size = block_size
        self.kernel_size = kernel_size
        self.layer_channels = layer_channels
        self.skip_channels = skip_channels
        self.aggregate_channels = aggregate_channels
        self.classes = classes
        self.residual_channels = residual_channels
        self.layer_nets = torch.nn.ModuleList()
        self.residual_connections = torch.nn.ModuleList()
        self.skip_connections = torch.nn.ModuleList()
        
        self.classes = classes

        self.aggregate1x1 = nn.Conv1d(
            self.skip_channels, self.aggregate_channels, 1)
        self.output1x1 = nn.Conv1d(self.aggregate_channels, self.classes, 1)
        self.embedding = nn.Conv1d(1, self.residual_channels, 1)

        for layer_index in range(self.layers):
            dilation = 2 ** (layer_index % self.block_size)
            self.layer_nets.append(WaveNetLayer(self.layer_channels, self.kernel_size, dilation))
            self.skip_connections.append(nn.Conv1d(self.layer_channels, self.skip_channels, 1))
            self.residual_connections.append(nn.Conv1d(self.layer_channels, self.residual_channels, 1))

    def forward(self, input: List[np.array]) -> (List[List[int]]):
        """ Take a tensor with shape (B, N)

        @param input (np.array): a np.array with shape (B, N)
        @param context (torch.Tensor): a tensor with shape (B, N, H)
        N: number of samples
        B: batch size
        H: context size

        @returns output (torch.Tensor): a variable/tensor of shape (B, N, output_size)
        """
        padded_source, lengths = pad_source(input)

        input_tensor = torch.tensor(padded_source, dtype=torch.float, device=self.device)
        output, _ = self.layers_forward(input_tensor)
        # masked_output = self.mask(output, lengths)
        return F.log_softmax(output, 1).permute(2, 0, 1,), lengths

    def layers_forward(self, layer_input: torch.Tensor, padding=True) -> torch.Tensor:
        layer_outputs = []
        layer_aggregate_input = torch.zeros(
                layer_input.size(0), self.skip_channels, layer_input.size(2), device=self.device)
        for layer_index in range(self.layers):
            layer_output = self.layer_nets[layer_index](
                layer_input, padding)
            layer_skip_output = self.skip_connections[layer_index](layer_output)
            layer_aggregate_input = layer_aggregate_input + layer_skip_output
            layer_input = layer_input + self.residual_connections[layer_index](layer_output)
            layer_outputs.append(layer_output)
        aggregate = self.aggregate1x1(F.relu(layer_aggregate_input))
        output = self.output1x1(F.relu(aggregate))
        return output, layer_outputs

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.output1x1.weight.device
