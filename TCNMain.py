import os
import argparse
from DataPre import *
from DataLoad import *
from Sample import *
from RuleClassfion import *
from Train import *
from model import TCN
from Test import *
from train_test_split import *

import pickle
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit


parser = argparse.ArgumentParser(description='TS')
parser.add_argument('--jobs', type=int, default=4)
parser.add_argument('--label_path', type=str, default='./datasets/data/train/train_labels.csv')
parser.add_argument('--trn_path',type=str,default='./datasets/data/train')
parser.add_argument('--test_path',type=str,default='./datasets/data/test')
parser.add_argument('--result_path',type=str,default='./work/prediction_result')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')


args = parser.parse_args()


if __name__=="__main__":

    input_channels = 1
    n_classes = 2
    channel_sizes = [6]*4
    kernel_size = 7
    dropout = 0.05

    model = TCN(input_channels, n_classes, channel_sizes, kernel_size, dropout)
    max_accuracy, max_precise, max_recall, max_f1_score = 0., 0., 0., 0.
    for epoch in range(1, args.epochs+1):
        model = Train(model,args=args)
        max_accuracy, max_precise, max_recall, max_f1_score = test(model,max_accuracy, max_precise, max_recall, max_f1_score )








