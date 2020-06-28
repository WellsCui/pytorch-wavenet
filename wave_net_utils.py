
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch
import torchvision.transforms.functional as TF
from typing import List, Tuple, Dict, Set, Union
from multiprocessing import Process, Queue
import re
import os
import librosa  # for audio processing
import time
import math


def load_voices_files(voice_files):
    voices = []
    for voice_file in voice_files:
        _, samples = wavfile.read(voice_file, True)
        voices.append(samples.tolist())
    return voices


def get_max_length(voices):
    max_length = 0
    for voice in voices:
        voice_length = len(voice)
        if voice_length > max_length:
            max_length = voice_length
    return max_length


def fill_voices_data_with_pads(voices):
    padded_voices = []
    lengths = []
    max_length = get_max_length(voices)
    for voice in voices:
        voice_length = len(voice)
        lengths.append(voice_length)
        padded_voices.append(voice+[0.0]*(max_length-voice_length))
    return padded_voices, lengths


def quantize_voices(voices: np.array, u: int):
    rs = np.array(voices)
    rs = np.sign(rs)*np.log(1+u*np.abs(voices))/np.log(1+u)
    return rs


def get_voices_labels(voices: np.array, label_count=256):
    Y = np.array(voices)/(65536/2)
    mu=label_count-1
    Y = np.sign(Y)*np.log(1+mu*np.abs(Y))/np.log(1+mu)
    step = 2 / label_count
    Y = Y // step + label_count // 2
    return Y.astype(int)

def read_corpus_from_LJSpeech(file_path, source, line_num=-1):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
        each line is begin with 11 charactors wav file name:  'LJ001-0004|'
        the rename text is the speech of the wav.                  
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    line_count = 0
    for line in open(file_path):
        sent_info = line.split('|')
        voice_name = sent_info[0]
        sent = re.sub('[,";:\?\(\)]', '', sent_info[-1])\
            .lower()\
            .replace("-- ", "")\
            .replace("-", " ")\
            .replace("'s ", " 's ")\
            .replace(". ", " ")\
            .strip()\
            .split(' ')
        last_char = sent[-1][-1]
        if last_char in ['.', ';', ","]:
            sent[-1] = sent[-1][:-1]
        #     sent = sent + [last_char]

        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        yield (voice_name, sent)
        line_count += 1
        if line_count == line_num:
            break


def load_train_data(voice_path: str, data_size: int, epoch_size: int, data_queue: Queue, clip_size=7680, repeat=1, decade_rate=1):
    print("loading train data ...")
    sample_rate = 22000
    resample_rate = 8000
    data = read_corpus_from_LJSpeech(
        voice_path + '/metadata.csv', 'tgt', data_size)
    voices = []
    corpus = []
    data_count = 0
    epoch_count = 0

    remaining_records = int((1 - decade_rate) * epoch_size // 1)
    print("remaining train data length:", remaining_records)
    corpus_map = []

    for voice_file, sent in data:
        corpus_map.append((voice_file, sent))
    corpus_index_array = list(range(len(corpus_map)))
    for rd in range(repeat):
        print("pushing new round train data:", rd)

        np.random.shuffle(corpus_index_array)
        for idx in corpus_index_array:
            voice_file, sent = corpus_map[idx]
            while not data_queue.empty():
                time.sleep(3)
            voice_file = voice_path+'/'+voice_file+'.wav'
            if not os.path.isfile(voice_file):
                continue
            _, samples = wavfile.read(voice_file, True)
            if len(samples) < clip_size:
                continue
            start_index = np.random.randint(0, len(samples)-clip_size)
            clip = samples[start_index:start_index+clip_size]

            # clip_count = len(samples) // clip_size
            # clips = np.split(samples[:clip_count*clip_size], clip_count)
            clip_count = 1      
            voices.append(clip)
            corpus.append(sent)
            epoch_count = epoch_count + clip_count
            data_count = data_count + clip_count
            if epoch_count >= epoch_size:
                # print("push new train data ...")
                # train_data = list(zip(voices, corpus))
                # labels = get_voices_labels(voices)
                data_queue.put(voices, True)
                voices = []
                corpus = []
                epoch_count = 0

    print("all train data has been loaded")
    data_queue.put(None, True)
    return


def batch_iter_to_queue(data_queue, batch_queue, loss_queue, epoch_num, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    print("geting train data ...")
    data = data_queue.get(True)

    train_index = 0
    while data is not None:
        train_index += 1
        print("start new training %d: data(size = %s) in %d epoches : ..." %
              (train_index, len(data), epoch_num))
        for epoch in range(epoch_num):
            # print("epoch:", epoch, "started")
            batch_num = math.ceil(len(data) / batch_size)
            index_array = list(range(len(data)))

            if shuffle:
                np.random.shuffle(index_array)
            loss_sum = 0.0

            for i in range(batch_num):
                indices = index_array[i * batch_size: (i + 1) * batch_size]
                batch_data = [data[idx] for idx in indices]
                batch_queue.put((epoch, np.array(batch_data)), True)
                loss_sum = loss_sum + loss_queue.get()
            if loss_sum/batch_num < 0.01:
                break

        # print("geting train data ...")
        data = data_queue.get(True)
        # if data is not None:
        #     print("recieved train data (size = %d) ..." % len(data))
        # else:
        #     print("recieved no train data")

    batch_queue.put((None, None), True)
