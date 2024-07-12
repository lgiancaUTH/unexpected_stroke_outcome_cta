import os
import sys

from ast import Lambda
from turtle import forward
import torch
torch.cuda.empty_cache()
import copy 
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F

from skimage.transform import resize 
from torch import randn, tensor as tensor 
from torch import clamp as clamp  
import warnings
warnings.filterwarnings("ignore")

depth = 4
activation = 'relu'
nFilters = 16

class NormLayer(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()

        self.minVar = tensor(0.001, requires_grad=True)
        self.maxVar = tensor(1., requires_grad=True)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.minVar = self.minVar.to(*args, **kwargs) 
        self.maxVar = self.maxVar.to(*args, **kwargs) 

        return self

    def forward(self, inputs):
        res = torch.clamp(inputs, self.minVar, self.maxVar)
        res = (res - self.minVar) / (self.maxVar-self.minVar)
        res = torch.clamp(res, 0, 1)
        return res


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):

        (batch, channel, t, h, w) = x.size()
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)

        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class leNet(nn.Module):
    def __init__(self, depth, activation, nFilters, merged):
        super().__init__()
        self.depth = depth 
        self.input_shape = 1
        if not merged:
            self.conv1 = nn.Conv3d(1, out_channels=nFilters, 
                                kernel_size=(3, 3, 3), padding='same')
        self.conv2 = []
        self.activation = nn.ReLU()
        self.maxpool = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        
        if self.depth > 1:
            for i in range(1, self.depth):
                self.conv2.append(nn.Conv3d(nFilters, out_channels=nFilters, 
                                    kernel_size=(3, 3, 3), padding='same'))
                self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

def leNetWithInputLayer(input_shape, **args):
    inputs = tensor(input_shape)
    outputs = leNet(**args)(inputs)
    return outputs 

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class vggNet(nn.Module):
    def __init__(self, depth, activation, nFilters, nConv, merged):
        super().__init__()

        self.depth = depth 
        self.nConv = nConv
        self.merged = merged

        self.conv2 = []

        self.input_shape = 1
        if merged:
            self.input_shape = nFilters

        for _ in range(0, self.depth):
            for _ in range(0, self.nConv):
                self.conv2.append(nn.Conv3d(self.input_shape, out_channels=nFilters, 
                                kernel_size=3, padding='same'))       
                self.conv2.append(nn.ReLU())
                self.conv2.append(nn.BatchNorm3d(nFilters))
                self.input_shape = nFilters

            self.conv2.append(MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)) 
                
        self.conv2 = nn.Sequential(*self.conv2) 

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 

        return self       

    def forward(self, x):

        x = self.conv2(x)

        return x

class torch_sNetAutoPreprocessingVggNetWithSkipConn(nn.Module):
    def __init__(self, n_classes, depthBefore, depthAfter, nFilters, nConv, addDenseLayerNeurons, last_fc):
        super().__init__()
        self.nclasses = n_classes
        self.adddense = addDenseLayerNeurons
        self.last_fc = last_fc
        self.base = vggNet(depth=depthBefore,
                            activation='relu',
                            nFilters=nFilters,
                            nConv=nConv, merged=False)
        self.nL = NormLayer('normLayer1')
        self.merged = vggNet(depth=depthAfter,
                            activation='relu',
                            nFilters=nFilters,
                            nConv=nConv, merged=True)
        self.nonSymnet = vggNet(depth=depthAfter,
                                activation='relu',
                                nFilters=nFilters,
                                nConv=nConv, merged=True)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=[1, 1, 1])    
        self.fc = nn.Linear(72, n_classes)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.nL = self.nL.to(*args, **kwargs) 
        self.base = self.base.to(*args, **kwargs) 
        
        return self

    def forward(self, x0, x1):
        input_left = x0
        input_right = x1


        normLeft = self.nL(input_left)
        normRight = self.nL(input_right)
        #print(f"-------norm right size: {normRight.size()}-----------")

        processed_left = self.base(normLeft)
        #print(f"-------processed left size: {processed_left.size()}-----------")

        processed_right = self.base(normRight)

        l1_dist = LambdaLayer(lambda tensors: torch.abs(tensors[0] - tensors[1])) #torch.cdist(processed_left, processed_right, p=2)
        merged_layer = l1_dist([processed_left, processed_right])
        #print(f"--------here is size of merge: {merged_layer.size()}------")

        nx = self.merged(merged_layer)
        #print(f"-------processed left size: {processed_left.size()}-----------")
        xLeft = self.nonSymnet(processed_left)
        xRight = self.nonSymnet(processed_right)

        nx = torch.cat((nx, xLeft, xRight), dim=1)
        nx = self.avgpool(nx)

        nx = nx.view(nx.size(0), -1) #this is for CLIP model, copy from resnet
        if self.last_fc:
            nx = self.fc(nx)

        return nx

if __name__ == '__main__':
    print('run well')

