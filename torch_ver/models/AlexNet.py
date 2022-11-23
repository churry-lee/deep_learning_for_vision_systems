import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F


# AlexNet Simple ver.
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # input image = (3, 32, 32)
        conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1, padding="same", dilation=1)
        conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding="same", dilation=1)
        conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding="same", dilation=1)

        pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        dropout1 = nn.Dropout2d(0.3)
        flatten = nn.Flatten()   # 4 * 4 * 64 = 1,024

        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(), pool,
            conv2, nn.ReLU(), pool,
            conv3, nn.ReLU(), pool,
            dropout1, flatten
        )

        fc1 = nn.Linear(4 * 4 * 64, 500)
        dropout2 = nn.Dropout2d(0.4)
        fc2 = nn.Linear(500, 10)

        self.fc_module = nn.Sequential(
            fc1, nn.ReLU(), dropout2,
            fc2
        )


    def forward(self, x):
        x = self.conv_module(x)
        x = self.fc_module(x)
        output = F.softmax(x, dim=1)
        return output

# alexnet = AlexNet()
# torchsummary.summary(alexnet, input_size=(3, 32, 32))
