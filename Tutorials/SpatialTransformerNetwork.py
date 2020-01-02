from __future__ import print_function
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='Data/MNIST', train=True, download=True, 
                    transform=transforms.Compose([
                        transforms.ToTensor(), 
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])), batch_size=64, shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='Data/MNIST', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=False, num_workers=0)
