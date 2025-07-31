"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

import os
import pickle
import string

import torch
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


class SubMNIST(Dataset):
    """
    Constructs a subset of MNIST dataset from a pickle file;
    expects pickle file to store list of indices
    """

    def __init__(self, path, aggregator_, mnist_data=None, mnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param mnist_data: MNIST dataset inputs 
        :param mnist_targets: MNIST dataset labels
        :param transform:

        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        self.aggregator_ = aggregator_

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])
        if self.aggregator_ == "centralized":
            if mnist_data is None or mnist_targets is None:
                self.data, self.targets = get_mnist()
            else:
                self.data, self.targets = mnist_data, mnist_targets
        else:
            if mnist_data is not None and mnist_targets is not None:
                self.data = torch.tensor(mnist_data, dtype=torch.float32)  
                self.targets = torch.tensor(mnist_targets, dtype=torch.int64)  
            else:
                raise NotImplementedError("Loading from file path is not implemented in this example.")

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        if self.aggregator_ == "centralized":
            img, target = self.data[index], self.targets[index]

            img = Image.fromarray(img.numpy())

            if self.transform is not None:
                img = self.transform(img)

            target = target

            return img, target, index
        
        else:
            # Directly return the data (embedding) and the corresponding target
            return self.data[index], self.targets[index], index


class SubCAMELYON17(Dataset):
    """
    Constructs a subset of Camelyon17 dataset from a pickle file;
    expects pickle file to store list of indices

    """

    def __init__(self, path):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Camelyon17 dataset inputs 
        :param cifar10_targets: Camelyon17 dataset labels 
        :param transform:

        """
        split_data = np.load(path)

        self.data = split_data['embeddings']
        self.targets = split_data['labels']
        self.data = torch.tensor(self.data, dtype=torch.float32)  
        self.targets = torch.tensor(self.targets, dtype=torch.int64)  

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        # Directly return the data (embedding) and the corresponding target
        return self.data[index], self.targets[index], index
        

def get_mnist():
    """
    gets full (both train and test) MNIST dataset inputs and labels;
    the dataset should be first downloaded

    :return:
        mnist_data, mnist_targets
    """
    mnist_path = os.path.join("data", "mnist", "dataset")
    assert os.path.isdir(mnist_path), "Download MNIST dataset!!"

    mnist_train =\
        MNIST(
            root=mnist_path,
            train=True, download=False
        )

    mnist_test =\
        MNIST(
            root=mnist_path,
            train=False,
            download=False)

    mnist_data = \
        torch.cat([
            torch.tensor(mnist_train.data),
            torch.tensor(mnist_test.data)
        ])

    mnist_targets = \
        torch.cat([
            torch.tensor(mnist_train.targets),
            torch.tensor(mnist_test.targets)
        ])

    return mnist_data, mnist_targets