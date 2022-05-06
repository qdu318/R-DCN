# encoding utf-8
import json
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
import utils
import csv
import numpy as np
import torch
import pandas as pd
from torchvision import datasets, transforms
from  DatasetFromCSV import DatasetFromCSV
import argparse





input_channels= 1
seq_length = int(7/ input_channels)
steps=0



dataset = DatasetFromCSV("./train_fs.csv")
train_dataset= torch.utils.data.DataLoader(dataset, batch_size=16)


def Train(model,args):
    global steps
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    train_loss = 0

    for  batch_idx, (parameters,lable) in enumerate(train_dataset):

        parameters = utils.to_var(parameters)
        parameters=parameters.view(-1,input_channels,seq_length)
        out = model(parameters)


        lable = lable.type(torch.long)
        loss = F.nll_loss(out, lable)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                args.epochs, batch_idx * 16, len(train_dataset.dataset),
                    100. * batch_idx / len(train_dataset), train_loss.item() / 100, steps))
            train_loss = 0
    return model