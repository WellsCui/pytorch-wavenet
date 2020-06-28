#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causal_conv1d import CausalConv1d


class WaveNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, context_size=0):
        """ Init CNN Model.

        @param embed_size (int): Word Embedding size 
        """
        super(WaveNetLayer, self).__init__()
        self.dilation = dilation
        # print("WaveNetLayer.dilation:", dilation)
        self.in_channels = in_channels
        self.output_channels = out_channels
        self.layerConv1d = CausalConv1d(
            in_channels, in_channels, kernel_size, stride=1, dilation=dilation)
        
        self.context_size = context_size
        # if context_size > 0:
        #     self.context_filter = CausalConv1d(
        #     out_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        #     self.context_gate = CausalConv1d(
        #     out_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        
    
    def forward(self, X: torch.Tensor, context: torch.Tensor, padding=True) -> torch.Tensor:
        """ Take a tensor with shape (B, C, N)

        @param X (torch.Tensor): a tensor with shape (B, C, N)
        @param Y (torch.Tensor): a tensor with shape (B, N, H)
        N: number of samples
        B: batch size
        C: input channel
        H: context size

        @returns output (torch.Tensor): a variable/tensor of shape (B, N, output_channels)
        """
        output = self.layerConv1d(X, padding)
        
        # if self.context_size > 0:
        #     filtered = filtered + self.context_filter(context, padding)
        #     gated = gated + self.context_gate(context, padding)
        
        output = torch.sigmoid(output[:, :self.output_channels, :]) * torch.tanh(output[:, self.output_channels:, :])
        return output
    
