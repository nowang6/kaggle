# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from nltk.corpus import stopwords
import jieba
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


MAX_VOCAB_SIZE = 120000  # 词表长度限制
unk = '<UNK>'
pad = '<PAD>'
MAX_DATA_FOR_DEBUG=1000

def load_stop_words():
    df = pd.read_csv("data/stop_words.txt")
    return set(df["word"])
    #return set(stopwords.words("english"))

def build_vocab_and_data(file_path):
    df = pd.read_csv(file_path,sep = '\t',warn_bad_lines=True, error_bad_lines=False)
    # df = pd.read_csv(file_path, sep = '\t', compression="zip")
    phrases = []
    stop_words = load_stop_words()

    phrase_max_len = 0

    k=0
    for phrase in tqdm(df['Phrase']):
        k += 1
        if k>MAX_DATA_FOR_DEBUG:
            break
        if (not isinstance(phrase, str)):
            phrase = ""
        phrase = phrase.lower()
        phrase = phrase.strip()
        phrase = phrase.replace(" ","")
        # english
        # p = ''.join([c for c in p if c not in punctuation])
        # reviews_split = p.split()
        # reviews_wo_stopwords = [word for word in reviews_split if not word in stopwords_set]
    
        #chinese
        words = jieba.lcut(phrase, HMM=True)
        if (phrase_max_len < len(words)):
            phrase_max_len = len(words)
        #words = list(phrase)
        words_without_stop = [word for word in words if not word in stop_words]
        phrases.append(words_without_stop)
    vocab = build_vocab_from_iterator(phrases, specials=[pad,unk]).get_itos()

    # encode and pad
    MAX_LEN = phrase_max_len + 5
    encoded_phrases = []
    for phrase in tqdm(phrases):
        encoded_one_phase = [vocab.index(pad)] * MAX_LEN # pad sentens to max len
        for i, word in enumerate(phrase):
            if word in vocab:
              encoded_one_phase[i]=vocab.index(word)
            else:
              encoded_one_phase[i]=vocab.index(unk)
        encoded_phrases.append(encoded_one_phase)
            

    # build target
    sentiments = []
    k = 0 
    for sentiment in tqdm(df['Sentiment']):
        k += 1
        if k>MAX_DATA_FOR_DEBUG:
            break
        if (not isinstance(sentiment, int)):
            print(f'Error, target is not int: {sentiment}')
        sentiments.append(sentiment)
    return vocab, encoded_phrases, sentiments
    
def init_network(model):
    for name, w in model.named_parameters():
        if "embedding" not in name: # 不初始化embeeding
            if 'weight' in name:
                nn.init.xavier_normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass




def print_data(vocab, train_data, train_target,index):
    for i in index:
        phrase = train_data[i]
        decoded_phrase = " ".join([vocab[j] for j in phrase])
        print(f'{i} phase: {decoded_phrase}')
        print(f'{i} target: {train_target[i]}')





if __name__ == "__main__":
    train_data = "data/THUCNews/train.txt"
    #train_data = "data/test.tsv"
    pretrain_dir = "data/THUCNews/sgns.sogou.char"
    batch_size = 1000
    vocab, train_data, train_target = build_vocab_and_data(train_data)
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_target, test_size=0.3)
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))


    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    

