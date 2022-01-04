#!/bin/bash
# Download CIFAR and pretrained models

curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --output CIFAR10/cifar-10-python.tar.gz
tar -xzvf CIFAR10/cifar-10-python.tar.gz CIFAR10/

git clone https://github.com/huyvnphan/PyTorch_CIFAR10 PyTorch_CIFAR10/

cd PyTorch_CIFAR10
python train.py --download_weights 1
cd ..