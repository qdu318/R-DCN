import numpy as np
import pandas as pd
from sklearn.utils import shuffle as reset


def train_test_split(data, test_size=0.3, shuffle=True, random_state=None):

    if shuffle:
        data = reset(data, random_state=random_state)

    train = data[int(len(data) * test_size):].reset_index(drop=True)

    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    return train, test
