import os
import time
import sys
import math
import argparse
import json
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from typing import List, Tuple, Dict, Set, Union
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from dataloader import DataLoader
from char_utils import CharacterDic

import pylab
import librosa


from wave_net import WaveNet
# from wave_net_utils import load_voices_files, get_max_length,fill_voices_data_with_pads, quantize_voices, get_voices_labels,load_train_data, batch_iter_to_queue


# device = torch.device("cuda:0")





def train(model_config, data_config, output_path, device,
          epoch_size, max_epoch, batch_size, repeats, 
          decade_rate, clip_grad, log_every, valid_every):
    print('use device: %s' % device, file=sys.stderr)
    model = WaveNet(**model_config)
    model.train()
    model = model.to(torch.device(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    data_loader = DataLoader(**data_config)
    batch_queue, loss_queue = data_loader.load_train_data(
        epoch_size, max_epoch, batch_size, repeats, decade_rate)
    dev_data = data_loader.load_dev_data()

    log_iter = 1
    char_dic = CharacterDic()
    criterion = nn.CrossEntropyLoss()
    epoch, voices, tgt_sents = batch_queue.get(True)
    while voices is not None and tgt_sents is not None:
        # train_iter += 1
        optimizer.zero_grad()
        # forward + backward + optimize
        # print("labels.shape:", labels.shape)
        print('getting target:')
        targets, target_lengths, = char_dic.str2Idx(tgt_sents)
        print('target_lengths:', target_lengths)
        targets_tensor = torch.tensor(targets, device=model.device)
        print('getting output:')
        outputs, input_lengths = model(voices)
        ctc_loss = nn.CTCLoss()
        print('getting loss:')
        
        print('input_lengths:', input_lengths)
        loss = ctc_loss(outputs, targets_tensor, torch.tensor(input_lengths), torch.tensor(target_lengths))
        loss.backward()
        print('loss:', loss.item())
        optimizer.step()  
        loss_queue.put(loss.item(), True)
        epoch, voices, tgt_sents = batch_queue.get(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    train(config["model_config"], config["data_config"],
          **config["train_config"])

