# Adapted from Meta-Learning with Differentiable Convex Optimization, CVPR 2019
# https://github.com/kjunelee/MetaOptNet

import torch.nn as nn
import math
from braincog.base.node import *

__all__ = ['SNN_ProtoNetEmbedding']



# Embedding network used in Matching Networks (Vinyals et al., NIPS 2016), Meta-LSTM (Ravi & Larochelle, ICLR 2017),
# MAML (w/ h_dim=z_dim=32) (Finn et al., ICML 2017), Prototypical Networks (Snell et al. NIPS 2017).

class SNN_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, config, retain_activation=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if retain_activation:
            # change the requires_thres_grad = True
            self.block.add_module("LIFNode", LIFNode(threshold=0.3))

        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self,x):
        out = self.block(x)
        return out

class SNN_ProtoNetEmbedding(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True, config=None):
        super().__init__()

        self.config = config

        self.encoder = nn.Sequential(
            SNN_ConvBlock(x_dim, h_dim, self.config),
            SNN_ConvBlock(h_dim, h_dim, self.config),
            SNN_ConvBlock(h_dim, h_dim, self.config),
            SNN_ConvBlock(h_dim, z_dim, self.config, retain_activation=retain_last_activation), 
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def reset_LIF(self):
        # 调用lif的n_reset函数
        for i in range(4):
            # print(type(self.encoder[i].block[2]).__name__)
            self.encoder[i].block[2].n_reset()

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)