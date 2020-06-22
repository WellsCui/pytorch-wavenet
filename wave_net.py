#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from causal_conv1d import CausalConv1d
from wave_net_layer import WaveNetLayer
from wave_net_utils import fill_voices_data_with_pads
from typing import List, Tuple, Dict, Set, Union, Optional
from multiprocessing import Process, Queue

class WaveNet(nn.Module):
    """ Simple CNN Model:
        - 1D CNN
    """

    def __init__(self, 
        layers=20, 
        block_size=10,
        kernel_size=2,
        layer_channels=16, 
        skip_channels=32,
        aggregate_channels=64,
        classes=256,
        conditional_features={
            "enabled": False,
            "channels": 0,
            "upsamp_window": 0,
            "upsamp_stride": 0,
        }):
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
        self.conditional_features = conditional_features
        self.aggregate_channels = aggregate_channels
        self.classes = classes
        self.layer_nets = torch.nn.ModuleList()
        self.skip_connections = torch.nn.ModuleList()
        
        self.classes = classes

        self.aggregate1x1 = nn.Conv1d(
            self.skip_channels, self.aggregate_channels, 1)
        self.output1x1 = nn.Conv1d(self.aggregate_channels, self.classes, 1)
        self.embedding = nn.Conv1d(1, self.layer_channels, 1)

        for layer_index in range(self.layers):
            dilation = 2 ** (layer_index % self.block_size)
            self.layer_nets.append(WaveNetLayer(
                self.layer_channels, self.layer_channels, self.kernel_size, dilation, context_size=self.conditional_features["channels"]))
            self.skip_connections.append(nn.Conv1d(self.layer_channels, self.skip_channels, 1))

    def forward(self, input: np.array, context: Optional[torch.Tensor]) -> (torch.Tensor, List[int]):
        """ Take a tensor with shape (B, N)

        @param input (np.array): a np.array with shape (B, N)
        @param context (torch.Tensor): a tensor with shape (B, N, H)
        N: number of samples
        B: batch size
        H: context size

        @returns output (torch.Tensor): a variable/tensor of shape (B, N, output_size)
        """
        input_tensor = torch.tensor(input, dtype=torch.float, device=self.device)
        layer_input = self.embedding(input_tensor.unsqueeze(1))
        output, _ = self.layers_forward(layer_input, context)
        # masked_output = self.masks(output, lengths)
        return F.log_softmax(output, 1)


    def layers_forward(self, layer_input: torch.Tensor, context: torch.Tensor, padding=True) -> torch.Tensor:
        layer_outputs = []
        layer_aggregate_input = torch.zeros(
                layer_input.size(0), self.skip_channels, layer_input.size(2), device=self.device)
        for layer_index in range(self.layers):
            layer_output = self.layer_nets[layer_index](
                layer_input, context, padding)
            layer_skip_output = self.skip_connections[layer_index](layer_input)
            layer_aggregate_input = layer_aggregate_input + layer_skip_output
            layer_input = layer_input + layer_output
            layer_outputs.append(layer_output)
        aggregate = self.aggregate1x1(F.relu(layer_aggregate_input))
        output = self.output1x1(F.relu(aggregate))
        return output, layer_outputs

    def reconstruct_from_output(self, source: torch.Tensor) -> List[List[int]]:
        """ reconstruct voice from forward output.

        @param source (torch.Tensor): a torch.Tensor with shape (B, C, N), where B = batch size,
                                     N = max source length, C = channel_size.
        @param source_lengths (int): A list of lengths of source
        """
        voices = []
        for e_id in range(source.size(0)):
            # voice = source[e_id, :, :src_len].cpu().detach().numpy()
            voice_softmax = source[e_id, :, :]
            Y = torch.argmax(voice_softmax, axis=0)-128
            Y = Y.float() / 128
            voice = torch.sign(Y)*(1/256)*((1+256)**torch.abs(Y)-1)
            voices.append(voice)
        return voices

    def generate(self, leading_samples: List[List[int]], context: torch.Tensor, sample_queue: Queue) -> List[List[int]]:
        """ generate voice with context.

         @param context (np.array): a np.array with shape (B, N, H)
                                    N: number of samples
                                    B: batch size
                                    H: context size

        """
        if leading_samples is not None:
            input_tensor = torch.tensor(leading_samples, dtype=torch.float, device=self.device)
            layer_inputs = self.embedding(input_tensor.unsqueeze(1))
            outputs, layer_outputs = self.layers_forward(layer_inputs, context)
            last_output = F.log_softmax(outputs[:,:,-1:], 1)
        sample_num = context.shape[1]
        batch_size = context.shape[0]
        samples = []
        receptive_fields = []
        for layer_index in range(self.layers):
            layer = self.layer_nets[layer_index]
            if layer_index == 0:
                receptive_fields_size = self.kernel_size
                sample = self.reconstruct_from_output(last_output)
                samples_tensor = torch.stack(sample)
                new_inpout = self.embedding(samples_tensor.unsqueeze(1))
                receptive_field = torch.cat((layer_inputs[:, :, -self.kernel_size+1:], new_inpout), dim=2)
                receptive_fields.append(receptive_field)
            else:
                receptive_fields_size = layer.dilation * \
                    (self.kernel_size-1) + 1
                receptive_fields.append(layer_outputs[layer_index-1][:, : ,-receptive_fields_size:])

        for sample_index in range(sample_num):
            layer_output_aggregate = torch.zeros(
                batch_size, self.skip_channels, 1, device=self.device)
            ctx = context[:, sample_index:sample_index+1, :]
            for layer_index in range(self.layers):
                layer_input = receptive_fields[layer_index]
                layer = self.layer_nets[layer_index]
                layer_output = layer(layer_input, ctx, padding=False)
                layer_skip_output = self.skip_connections[layer_index](layer_output)
                layer_output_aggregate = layer_output_aggregate + layer_skip_output
                if layer_index < self.layers-1:
                    layer_output = layer_output + layer_input[:, :, -1:]
                    receptive_fields[layer_index+1] = torch.cat(
                        [receptive_fields[layer_index+1][:, :, 1:], layer_output], dim=2)
            aggregate = self.aggregate1x1(F.relu(layer_output_aggregate))
            output = self.output1x1(F.relu(aggregate))
            output = F.log_softmax(output, 1)
            sample = self.reconstruct_from_output(output)
            samples_tensor = torch.stack(sample)
            new_inpout = self.embedding(samples_tensor.unsqueeze(1))
            receptive_fields[0] = torch.cat(
                [receptive_fields[0][:, :, 1:], new_inpout], dim=2)
            sample_queue.put(samples_tensor.cpu().detach().numpy(), True)
        sample_queue.put(None, True)
        print("generation done!")


    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.output1x1.weight.device
