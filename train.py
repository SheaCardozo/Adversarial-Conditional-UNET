import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt

from random import randint

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

BATCH_SIZE = 256

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")