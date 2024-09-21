import math
from functools import partial
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_cfg=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        build_norm_layer(dict(norm_cfg), out_channels)[1],
        nn.ReLU(inplace=True),
    )

    return m
    

class SparseBasicBlock(spconv.SparseModule):

    def __init__(self, inplanes, planes, stride=1, norm_cfg=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key),
            build_norm_layer(dict(norm_cfg), planes)[1],
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key),
            build_norm_layer(dict(norm_cfg), planes)[1],
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.net(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out