# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import random

parser = argparse.ArgumentParser(description='English Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, TextRCNN, DPCNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = './datasets'
    embedding = './datasets/glove.6B.300d.npz'
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model

    x = import_module(model_name)
    config = x.Config(dataset, embedding)
    random.seed(1)

    np.random.seed(1)

    torch.manual_seed(1)

    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    
    for param in model.embedding.parameters():
        param.requires_grad = False

    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)