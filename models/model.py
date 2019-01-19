import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.autograd import Variable

num_epochs = 10
num_classes = 5
learning_rate = 0.001

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        NUM_CLASSES = 5
        
        # input shape: [3, 140, 250]
        # conv --> relu
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)


        
        
        # input shape: [16, 140, 250]
        # conv --> relu --> max pool
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        
        # input shape: [16, 70, 125]
        # conv --> relu --> max pool
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), stride=1, padding=0)
#         self.relu3 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=5)
        
        # input shape: [16, 14, 25]
        self.linear = nn.Linear(30208, NUM_CLASSES)
        # minibatch size = 10
#         self.output = nn.Linear(10, 1)
        
    def forward(self, x):

        # print(x.shape)
        x = self.conv1(x)
#         print(x.shape)
        x = self.relu1(x)
#         print(x.shape)
        x = self.pool1(x)
#         print(x.shape)

        
        x = self.conv2(x)
#         print(x.shape)
        x = self.relu2(x)
#         print(x.shape)
        x = self.pool2(x)
#         print(x.shape)
        
        
        # flatten
        x = x.view(x.shape[0], -1)
#         print(x.shape)
        output = self.linear(x)
#         print(output.shape)
        
        return output