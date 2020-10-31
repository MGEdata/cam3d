import torch
import torch.nn.functional as F
from torch import nn


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
