#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6x6 is the size of the last convolutional feature map
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        print("self.fc3.bias.shape:")
        print(self.fc3.bias.shape)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        shape = x.shape[1:]
        num_features = 1
        for i in shape:
            num_features *= i

        return num_features


if __name__ == "__main__":
    net = Net()
    print("net:")
    print(net)
    print("================================")
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    print("optimizer:")
    print(optimizer)
    print("================================")
    params = list(net.parameters())
    print("len(parameters):")
    print(len(params))
    print("================================")
    print("params[0].size():")
    print(params[0].size())
    print("params[1].size():")
    print(params[1].size())
    print("params[2].size():")
    print(params[2].size())
    print("params[-1].size():")
    print(params[-1].size())
    print("================================")

    loss_func = nn.MSELoss()

    # in the training loop
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    input = torch.randn(1, 1, 32, 32)
    print("input:")
    print(input)

    # ground truth for the input
    target = torch.randn(10)
    target = target.view(1, -1)

    output = net(input)
    print("output:")
    print(output)
    print("================================")

    loss = loss_func(output, target)
    print("loss:")
    print(loss)

    # zero gradients before backward
    optimizer.zero_grad()

    print("================================")
    loss.backward()
    print("params[1].grad:")
    print(params[1].grad)

    print("================================")
    optimizer.step()  # update the parameters
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
