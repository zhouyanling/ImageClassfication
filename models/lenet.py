#-*-coding:utf-8-*-
#Lenet-5 model

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LeNet(nn.Module):
    def __init__(self,num_class):
        super(LeNet,self).__init__()
        #input[N,1,28,28]
        #self.conv1 = nn.Conv2d(1,6,5)  # gray image
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.avepool = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        #flatten[N,400]
        #self.fc1 = nn.Linear(400,120)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_class)
    def forward(self,x):
        str0 = x
        x = F.relu(self.conv1(x))
        x = self.avepool(x)
        x = F.relu(self.conv2(x))
        x = self.avepool(x)
        # (64,16,53,53)
        x = x.view((-1,44944))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.softmax(self.fc3(x), dim=1)
        x = self.fc3(x)
        return x