import numpy as np
import pickle

from PyTorch_CIFAR10.cifar10_models.vgg import vgg19_bn
from PyTorch_CIFAR10.cifar10_models.resnet import resnet50
from PyTorch_CIFAR10.cifar10_models.densenet import densenet169
from PyTorch_CIFAR10.cifar10_models.mobilenetv2 import mobilenet_v2
from PyTorch_CIFAR10.cifar10_models.googlenet import googlenet
from PyTorch_CIFAR10.cifar10_models.inception import inception_v3

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_cifar(folder="CIFAR10/cifar-10-batches-py"):
    X = None
    y = None
    for i in range(1, 6):
        df = unpickle(folder+f"/data_batch_{i}")

        if X is None:
            X = df[b'data']
        else:
            X = np.vstack([X, df[b'data']])
        
        if y is None:
            y = df[b'labels']
        else:
            y.extend(df[b'labels'])
            
    df = unpickle(folder+f"/test_batch")
    test_X = df[b'data']
    test_y = df[b'labels']

    return X, y, test_X, test_y

def get_model_sets(device):
    train_mods = []
    test_mods = []

    for m in [resnet50, densenet169, inception_v3, mobilenet_v2]:
        train_mods.append(m(pretrained=True).to(device))
        train_mods[-1].eval()
        
    for m in [vgg19_bn, googlenet]:
        test_mods.append(m(pretrained=True).to(device))
        test_mods[-1].eval()

    return train_mods, test_mods
