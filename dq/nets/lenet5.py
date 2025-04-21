import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder
import torch
import os
from pathlib import Path
import torchvision
from torchvision import datasets, transforms
import time


class LeNet5(nn.Module):
    def __init__(self, channel=1, num_classes=10, record_embedding: bool = False, no_grad: bool = False):
        super(LeNet5, self).__init__()
        self.embDim = 84

        # Layer 1: Conv 1x32x32 -> 6x28x28, kernel=5x5, stride=1
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=6, kernel_size=5, stride=1)
        # Layer 2: MaxPool 6x28x28 -> 6x14x14, kernel=2x2, stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Layer 3: Conv 6x14x14 -> 16x10x10, kernel=5x5, stride=1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # Layer 4: MaxPool 16x10x10 -> 16x5x5, kernel=2x2, stride=2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Layer 5: Flatten and FC 16x5x5=400 -> 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Layer 6: FC 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # Layer 7: FC 84 -> num_classes
        self.linear = nn.Linear(84, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.linear

    def get_embedding_dim(self):
        return self.embDim

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.embedding_recorder(x)
            x = self.linear(x)
        return x


def LeNet(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False, 
         pretrained: bool = False):
    if pretrained:
        # Check if the pretrained model exists
        pretrained_path = os.path.join('/kaggle/working', 'pretrained_lenet5_mnist.pth')
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained model from {pretrained_path}")
            model = LeNet5(channel=channel, num_classes=num_classes, record_embedding=record_embedding, no_grad=no_grad)
            model.load_state_dict(torch.load(pretrained_path))
        else:
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

        return model
    
    elif channel == 1 and (im_size[0] == 28 or im_size[0] == 32) and (im_size[1] == 28 or im_size[1] == 32):
        return LeNet5(channel=channel, num_classes=num_classes, 
                     record_embedding=record_embedding, no_grad=no_grad)
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")


def LeNet5Model(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
    return LeNet(channel, num_classes, im_size, record_embedding, no_grad, pretrained)
