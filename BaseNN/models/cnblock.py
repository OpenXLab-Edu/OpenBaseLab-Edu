from turtle import forward
from pyparsing import Forward
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

class CNBlock(nn.Module):
    def __init__(self,
                in_dim,
                out_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=dict(type='SyncBN'),
                ac_cfg=dict(type='ReLU'),
                groups=0,
                dilation=0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = norm_cfg
        self.ac = ac_cfg
        self.group = groups
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cnblock = ConvModule(self.in_dim,
                                self.out_dim,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                norm_cfg = self.norm,
                                ac_cfg = self.ac)
    
    def forward(self, input):
        res = self.cnblock(input)

        return res
