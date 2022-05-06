import numpy as np
import pandas as pd
from PIL import Image
from torch import optim, nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import tensorflow as tf
import torch



class DatasetFromCSV(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.data = pd.read_csv(csv_path,usecols=["Num","if_off","v_diff1","a_min5","a_mean5","a_max3","key_std","v_diff3_bin","Label"])
        self.labels = np.asarray(self.data.iloc[:, -1]).astype(float)
        self.transforms = transforms

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        img_as_np = np.asarray(self.data.iloc[index][1:-1]).astype(float)

        img_as_tensor = torch.Tensor(img_as_np)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)





class DatasetFromCSV2(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.data = pd.read_csv(csv_path,usecols=["Num","if_off","v_diff1","a_min5","a_mean5","a_max3","key_std","v_diff3_bin"])
        self.num = np.asarray(self.data.iloc[:, 0]).astype(float)
        self.transforms = transforms

    def __getitem__(self, index):

        ts_num = self.num[index]
        ts_as_np = np.asarray(self.data.iloc[index][1:]).astype(float)
        ts_as_tensor = torch.Tensor(ts_as_np)
        return (ts_as_tensor,ts_num)

    def __len__(self):
        return len(self.data.index)


