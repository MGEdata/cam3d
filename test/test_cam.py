# -*- coding: utf-8 -*-

# @Time    : 2020/11/4 16:04
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from itertools import combinations_with_replacement
import torch.nn.functional as F
import numpy as np
import os
import torch
from numpy import random
from torch import nn
from torch.nn import Module
from torch.utils import tensorboard

from cams.cam3d import GradCAM3dpp, GradCAM3d
from cams.nnn import Indexes


class Moudle1(Module):
    def __init__(self, *args):  # 鍒濆鍖?
        super(Moudle1, self).__init__()

        D_in, dens_out = 1, 22
        D1, D2 = 6, 1
        dense1, dense2 = 27, 64
        AvgPool3d_x, AvgPool3d_y, AvgPool3d_z =10,10,10
        self.link = D2 * AvgPool3d_x * AvgPool3d_y * AvgPool3d_x

        model_conv = nn.Sequential(
            # Indexes(D_in, D2,(10,10,10)),
            nn.Conv3d(D_in, D2, 1, stride=1, padding=0),
            # nn.BatchNorm3d(D2),
            # nn.ReLU(True),
            # nn.MaxPool3d(3, stride=1, padding=1),
            # nn.Dropout3d()
        )
        model_sigmod = nn.Sigmoid()
        model_Linear = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.link, dens_out),
            nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(dens_out, dens_out),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(dense2, dens_out),

        )

        self.model_conv = model_conv
        self.model_sigmod = model_sigmod
        self.avgpool = nn.AdaptiveAvgPool3d((AvgPool3d_x, AvgPool3d_y, AvgPool3d_z))
        self.model_Linear = model_Linear

    def forward(self, x, t=1):
        if t == 0:
            x = self.model_conv(x)
            print("conv out", x.shape)
            x = self.model_sigmod(x)

            x = self.avgpool(x)
            print("avgpool", x.shape)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            print("flatten", x.shape)
            x = self.model_Linear(x)
            print("linear", x.shape)
        else:
            x = self.model_conv(x)
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            x = self.model_Linear(x)

        return x


def run(train, test=None):
    if test is None:
        test = train
    train_x, train_y= train
    model = Moudle1()
    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 鍏锋湁閫氱敤浼樺寲绠楁硶鐨勪紭鍖栧寘锛屽SGD,Adam
    #
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')  # 涓昏鏄敤鏉ュ垽瀹氬疄闄呯殑杈撳嚭涓庢湡鏈涚殑杈撳嚭鐨勬帴杩戠▼搴?
    # loss_fn = torch.nn.MSELoss(reduction='mean')  # 涓昏鏄敤鏉ュ垽瀹氬疄闄呯殑杈撳嚭涓庢湡鏈涚殑杈撳嚭鐨勬帴杩戠▼搴?


    for t in range(20000):
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        y_pred = model(train_x, t)
        # y_pred = y_pred*we
        # prob = F.softmax(y_pred, dim=1)
        # prob = F.relu(y_pred)
        # _, idx = torch.max(prob, dim=1)
        loss = loss_fn(y_pred,train_y)
        if loss.item() < 0.001:
            break
        # if t % 10 == 9:
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if t%50==0:
        writer.add_scalar('loss',  loss.item(), global_step=t)


    test_x, test_y = test
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    y_pred = model(test_x)
    loss2 = loss_fn(y_pred, test_y)
    print(loss2.item())
    writer.close()
    return model


random.seed(0)
torch.random.manual_seed(0)


def get():
    x = random.random((120, 10, 10, 10)) + 0.00001

    # key = np.full((3,3,3),0.5)
    # key[1,1,1]=1.0
    # iter = list(combinations_with_replacement(range(8), 3))
    # y = []
    # for ai, index in enumerate(iter):
    #     i, j, k = index
    #     print(ai, index)
    #     x[ai, i:i + 3, j:j + 3, k:k + 3] = key
    #     # x[ai, i:i + 3, j:j + 3, k:k + 3] = x[ai, i:i + 3, j:j + 3, k:k + 3] + key
    #     l1, l2, l3 = random.randint(0, 8, 3)
    #     x[ai, l1:l1 + 3, l2:l2 + 3, l3:l3 + 3] = x[ai, l1:l1 + 3, l2:l2 + 3, l3:l3 + 3] + key
    #     # y.append((i ** 2 + j ** 2 + k ** 2) ** 0.5)
    #     y.append((i + j + k))


    iter = list(combinations_with_replacement(range(1,9), 3))
    y = []
    for ai, index in enumerate(iter):
        i, j, k = index
        print(ai, index)
        x[ai, i, j, k] = 1.0
        # x[ai, i:i + 3, j:j + 3, k:k + 3] = x[ai, i:i + 3, j:j + 3, k:k + 3] + key
        l1, l2, l3 = random.randint(1, 9, 3)
        x[ai, l1, l2, l3] = 1.0
        # y.append((i ** 2 + j ** 2 + k ** 2) ** 0.5)
        y.append((i + j + k-3))
    x = torch.tensor(x)
    x = x.unsqueeze(dim=1)
    y = torch.tensor(y).reshape((-1, 1))
    x = x.type(torch.float32)
    y = y.type(torch.float32)
    x = x / torch.max(x)
    return x, y


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


writer = tensorboard.SummaryWriter(log_dir="/home/iap13/wcx/tb/exp1", flush_secs=10)

data = [get() for i in range(10)]
x, y = zip(*data)

x = torch.cat(x, dim=0)
y = torch.cat(y, dim=0)

y_ = torch.zeros((1200, 22))
y = y.type(torch.long).squeeze()
y_ = torch.index_fill(y_, 1, y, torch.tensor(1))

# model = run((x, y), None)
# torch.save(model.state_dict(), "model_dict")

model = Moudle1()
model.load_state_dict(torch.load("model_dict"))

device = torch.device('cpu')
model.to(device)
model.eval()
target_layer = model.model_conv[-1]
# wrapped_model = GradCAM3d(model, target_layer)


wrapped_model = GradCAM3dpp(model, target_layer)
# wrapped_model = SmoothGradCAMpp(model, target_layer)

x = x.to(device)
y = y.to(device)

# for i in range(0, 1):
#     xi = x[i]
#     yi = y[i]
#
#     tensor_shown = xi.unsqueeze(0)
#
#     cams, idx = wrapped_model.forward(tensor_shown)
#     cams = cams.squeeze().cpu().numpy()
#     xi = xi.squeeze().cpu().numpy()
#     for t in range(10):
#         writer.add_images('countdown%d'%i,
#                          cams[t],
#                           global_step=t,
#                          dataformats='HW')
#         writer.close()
i=2
xi = x[i]
yi = y[i]

tensor_shown = xi.unsqueeze(0)

cams, idx = wrapped_model.forward(tensor_shown)
cams = cams.squeeze().cpu().numpy()
xi = xi.squeeze().cpu().numpy()
for t in range(10):
    writer.add_images('countdown%d'%i,
                     cams[t],
                      global_step=t,
                     dataformats='HW')
    writer.close()

# model = Moudle1()
# writer.add_graph(model.eval(),x)
# writer.close()