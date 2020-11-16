# -*- coding: utf-8 -*-

# @Time    : 2020/11/2 17:49
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import os.path as path
import re

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from skimage.io import concatenate_images, imshow, imsave
from torch import nn
from torch.nn import Module
from torchvision import transforms, models
import matplotlib.pyplot as plt

from statistics import mode, mean

import torch
import torch.nn.functional as F

# if __name__=="__main__":
#
#     import torch
#     import torch.nn. functional as F
#     from matplotlib.pyplot import imshow
#     from torchvision import models, transforms
#     import os
#
#     os.chdir(r"C:\Users\Administrator\Desktop\fig")
#     from  skimage import io
#     image = io.imread('dog.png')
#
#     imshow(image)
#
#     preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     ])
#     # convert image to tensor
#     tensor = preprocess(image)
#     tensor = normalize(tensor)
#     # reshape 4D tensor (N, C, H, W)
#     tensor = tensor.unsqueeze(0)
#
#     model = models.resnet18(pretrained=True)
#     model.eval()
#     print(model)
#
#     # In [7]:
#     # # the target layer you want to visualize
#     target_layer = model.layer4[1].conv2
#     #
#     # # wrapper for class activation mapping. Choose one of the following.
#     # wrapped_model = CAM(model, target_layer)
#     # wrapped_model = GradCAM(model, target_layer)
#     wrapped_model = GradCAMpp(model, target_layer)
#     # wrapped_model = SmoothGradCAMpp(model, target_layer, n_samples=5, stdev_spread=0.15)
#     # In [8]:
#     cam, idx = wrapped_model(tensor)
#
#     # # visualize only cam
#     imshow(cam.squeeze().numpy(), alpha=0.5, cmap='jet')
#
#     # In [11]:
#     # # reverse normalization for display
#     img = normalize.inverse_transform(tensor)
#     # In [12]:
#     heatmap = visualize(img, cam)
#     # In [13]:
#     # # save image
#     ##save_image(heatmap, './sample/{}_cam.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
#     # # save_image(heatmap, './sample/{}_gradcam.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
#     # # save_image(heatmap, './sample/{}_gradcampp.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
#     # save_image(heatmap, './sample/{}_smoothgradcampp.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
#     # In [14]:
#     # # or visualize on Jupyter
#     hm = (heatmap.squeeze().numpy().transpose(1, 2, 0))
#     imshow(hm)
from cams.cam3d import GradCAM, CAM, GradCAMpp, SmoothGradCAMpp
from cams.propressing.electro import ChgCar
from cams.ram import GradRAM, GradRAMpp, SmoothGradRAMpp
from cams.ram3d import GradRAM3d, GradRAM3dpp, SmoothGradRAM3dpp

torch.manual_seed(1)


class Moudle2(Module):
    def __init__(self, *args):  # 鍒濆鍖?
        super(Moudle2, self).__init__()

        D_in, dens_out = 1, 1
        D1, D2 = 20, 20
        dense1, dense2, dense3 = 400, 200, 200
        AvgPool3d_x, AvgPool3d_y = 2, 2
        self.link = D2 * AvgPool3d_x * AvgPool3d_y

        model_conv = nn.Sequential(
            nn.Conv2d(D_in, D1, 3, stride=1, padding=1),
            nn.BatchNorm2d(D1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(D1, D2, 3, stride=1, padding=1),
            nn.BatchNorm2d(D2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
        )

        model_Linear = nn.Sequential(
            nn.Linear(self.link, dense1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense2, dense3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense3, dens_out),
        )

        self.model_conv = model_conv
        self.avgpool = nn.AdaptiveAvgPool2d((AvgPool3d_x, AvgPool3d_y))
        self.model_Linear = model_Linear

    def forward(self, x, t=1):
        if t == 0:
            x = self.model_conv(x)
            print("conv out", x.shape)
            x = self.avgpool(x)
            print("avgpool", x.shape)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            print("flatten",x.shape)
            x = self.model_Linear(x)
            print("linear", x.shape)
        else:
            x = self.model_conv(x)
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            x = self.model_Linear(x)

        return x

class Moudle1(Module):
    def __init__(self, *args):  # 鍒濆鍖?
        super(Moudle1, self).__init__()

        D_in, dens_out = 1, 1
        D1, D2 = 20, 20
        dense1, dense2, dense3 = 400, 200, 200
        AvgPool3d_x, AvgPool3d_y,AvgPool3d_z = 2, 2, 2
        self.link = D2 * AvgPool3d_x * AvgPool3d_y*AvgPool3d_x

        model_conv = nn.Sequential(
            nn.Conv3d(D_in, D1, 3, stride=1, padding=1),
            nn.BatchNorm3d(D1),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(D1, D2, 3, stride=1, padding=1),
            nn.BatchNorm3d(D2),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=1),
        )

        model_Linear = nn.Sequential(
            nn.Linear(self.link, dense1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense1, dense2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense2, dense3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(dense3, dens_out),
        )

        self.model_conv = model_conv
        self.avgpool = nn.AdaptiveAvgPool3d((AvgPool3d_x, AvgPool3d_y, AvgPool3d_z))
        self.model_Linear = model_Linear

    def forward(self, x, t=1):
        if t == 0:
            x = self.model_conv(x)
            print("conv out", x.shape)
            x = self.avgpool(x)
            print("avgpool", x.shape)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            print("flatten",x.shape)
            x = self.model_Linear(x)
            print("linear", x.shape)
        else:
            x = self.model_conv(x)
            x = self.avgpool(x)
            x = torch.flatten(x, start_dim=1, end_dim=-1)
            x = self.model_Linear(x)

        return x

def run(train, test):
    train_x, train_y = train
    model = Moudle2()
    device = torch.device('cuda:0')
    model.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 鍏锋湁閫氱敤浼樺寲绠楁硶鐨勪紭鍖栧寘锛屽SGD,Adam

    # loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')  # 涓昏鏄敤鏉ュ垽瀹氬疄闄呯殑杈撳嚭涓庢湡鏈涚殑杈撳嚭鐨勬帴杩戠▼搴?
    loss_fn = torch.nn.MSELoss(reduction='mean')  # 涓昏鏄敤鏉ュ垽瀹氬疄闄呯殑杈撳嚭涓庢湡鏈涚殑杈撳嚭鐨勬帴杩戠▼搴?

    for t in range(50):
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        y_pred = model(train_x, t)
        loss = loss_fn(y_pred, train_y)

        # if t % 10 == 9:
        print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_x, test_y = test
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    y_pred = model(test_x)
    loss2 = loss_fn(y_pred, test_y)
    print(loss2.item())

    return model


if __name__=="__main__":
    ""

    import torch
    import torch.nn. functional as F
    import os
    torch.manual_seed(1)
    # model = models.vgg16(pretrained=True)
    # model.eval()
    # print(model)
    # weight_fc = list(
    #     model._modules.get('classifier').parameters())[-2].to('cpu').data

    os.chdir(r"/home/iap13/wcx/cx_flies/0")
    chgcar = ChgCar.from_file(r'/home/iap13/wcx/cx_flies/0/CHGCAR')
    data = chgcar.data["total"]
    image = data[::4,::4,::4]
    image=image.astype(np.float32)

    preprocess = transforms.Compose([
    torch.from_numpy,
    ])
    # convert image to tensor
    tensor1 = preprocess(image)
    # reshape 5D tensor (N, C, H, W, T)
    tensor1 = tensor1.unsqueeze(0)

    train_x = torch.rand((50,1,50,50))
    test_x = torch.rand(20,1,50,50)

    # train_y = torch.randint(0,2,(50,))
    # test_y = torch.randint(low=0,high=2,size=(20,))

    train_y = torch.rand(50,1)
    test_y = torch.rand(20,1)
    # #
    model = run((train_x, train_y), (test_x, test_y))

    torch.save(model.state_dict(), "model_dict")

    model = Moudle2()

    model.load_state_dict(torch.load("model_dict"))

    ####################################
    # model.to("cpu")
    # model.eval()
    #
    # data_result = []
    # for i in range(10):
    #     k = test_x[i]
    #     j = test_y[i]
    #
    #     tensor_shown = k.unsqueeze(0)
    #     score = model(tensor_shown)
    #     prob = F.softmax(score, dim=1)
    #     prob2 = prob
    #     prob, idx2 = torch.max(prob, dim=1)
    #     idx2 = idx2.item()
    #     prob = prob.item()
    #     print("{} predicted class ids {}\t probability {},{}".format(i, idx2, prob, prob2))
    #     data_result.append([i, j, idx2, prob])
    # data_result = np.array(data_result)
    #
    ####################################
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    target_layer = model.model_conv[-1]
    # wrapped_model = GradRAM(model, target_layer)
    # wrapped_model = GradRAMpp(model, target_layer)
    wrapped_model = SmoothGradRAMpp(model, target_layer)
    for i in range(10):
        x = test_x[i]
        y = test_y[i]
        tensor_shown = x.unsqueeze(0)

        x = x.to(device)
        y = y.to(device)

        cams, idx = wrapped_model.forward(tensor_shown)
        cams = cams.squeeze().cpu().numpy()



