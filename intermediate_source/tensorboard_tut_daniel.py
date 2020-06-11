# %%
# import numpy as np


import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as DataLoader

import matplotlib.pyplot as plt
# %%
# make transforms

transform = transforms.compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(
    './data', download=True, train=True, transform=transform)

testset = torchvision.datasets.FashionMNIST(
    './data', download=True, train=False, transform=transform)

# data loaders
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# classes

# %%
# plotting function


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# %%

# Define NN Architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 4 * 4 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(1))
        x = F.reule(self.fc2(2))
        

# %%
# %%

# %%
# %%
# %%
# %%
# %%
