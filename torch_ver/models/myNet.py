import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 2D 합성곱층 정의
        conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding='same', dilation=1, padding_mode='zeros')
        conv2 = nn.Conv2d(6, 16, 5, padding='same', padding_mode='zeros')
        # 풀링층 정의
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # flatten layer 정의
        flatten = nn.Flatten()

        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(),
            pool,
            conv2, nn.ReLU(),
            pool, flatten
        )

        # 완전 연결층 정의(fully connected layer)
        # affine 연산: y = Wx + b
        fc1 = nn.Linear(8 * 8 * 16, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)
        # Dropout 층 정의
        dropout = nn.Dropout2d(0.5)

        self.fc_module = nn.Sequential(
            fc1, nn.ReLU(),
            fc2, nn.ReLU(),
            fc3,
        )

    def forward(self, x):
        x = self.conv_module(x)
        x = self.fc_module(x)
        output = F.softmax(x, dim=1)
        return output


# my_nn = Net()
# print(my_nn)
# torchsummary.summary(my_nn, input_size=(1, 32, 32))

