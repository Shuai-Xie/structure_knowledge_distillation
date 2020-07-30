import sys

sys.path.insert(0, '/nfs/xs/tmp/structure_knowledge_distillation')

import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable

affine_par = True
import functools
import sys, os

# from libs import InPlaceABN, InPlaceABNSync

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')  # activation 对应 relu 函数，与 nn.BatchNorm2d 差别很大
BatchNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(BasicBlock, self).__init__()
        dilation = dilation * multi_grid
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            # InPlaceABNSync(out_features), # todo
            BatchNorm2d(out_features),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))  # feature 下采样到 很小尺寸
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        # bn = InPlaceABNSync(out_features) # todo
        bn = nn.Sequential([
            BatchNorm2d(out_features),
            nn.LeakyReLU(),
        ])
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # 各阶段的 feats, concat 到 (h,w), 再与原始特征 concat，最后 bottleneck 到 512D
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()

        # 3个 3x3 替换 1个 7x7
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)  # 1/2

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ceil_mode: use `ceil` instead of `floor`, 128->129
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # 1/4

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 1/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

        if layers == [3, 4, 23, 3]:  # resnet101
            self.pspmodule = PSPModule(2048, 512)  # 1/8
            self.head = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            # 中间监督, layer3 输出
            self.dsn = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                # InPlaceABNSync(512),  # todo
                BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        elif layers == [2, 2, 2, 2]:  # resnet18
            self.pspmodule = PSPModule(512, 128)
            self.head = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            self.dsn = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                # InPlaceABNSync(128), # todo
                BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        else:
            raise ValueError('layers should be [3, 4, 23, 3] or [2, 2, 2, 2]')

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # 1/2
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)  # 1/4
        x1 = self.layer1(x)  #
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)
        x_dsn = self.dsn(x3)  # 中间输出结果
        x4 = self.layer4(x3)
        # x = self.head(x4)  # baseline, 没有 PSP module
        x_feat_after_psp = self.pspmodule(x4)
        x = self.head(x_feat_after_psp)
        return [x, x_dsn, x_feat_after_psp, x4, x3, x2, x1]


def Res_pspnet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=21):
    '''
    ResNet(Bottleneck, [3, 4, 23, 3], num_classes) 101
    ResNet(BasicBlock, [2, 2, 2, 2], num_classes) 18
	'''
    model = ResNet(block, layers, num_classes)
    return model


if __name__ == '__main__':
    # os.environ["CUDA_HOME"] = "/nfs/xs/local/cuda-10.2"
    # 仍然使用了bashrc 指定的 nvcc

    img = torch.rand(1, 3, 512, 512)
    model = Res_pspnet(Bottleneck)
    model.eval()
    model(img)
