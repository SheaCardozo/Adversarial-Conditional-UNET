#!/bin/bash
# Download CIFAR and pretrained models

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
mkdir CIFAR10
tar -xzvf cifar-10-python.tar.gz -C CIFAR10/

git clone https://github.com/SheaCardozo/PyTorch_CIFAR10 PyTorch_CIFAR10/

cd PyTorch_CIFAR10
python train.py --download_weights 1
cd ..