
__all__ = ['HebbLinear', 'PlasticLinear', 'Linear','STDPLinear']

from ..plastic import *
from .activation import *

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class PlasticLinear(nn.Module):

    def __init__(self, ind, outd, scale=1, fan_in=False):
        super().__init__()
        self.w = PlasticParam(torch.empty(ind, outd))
        self.b = PlasticParam(torch.empty(outd))
        if fan_in:
            self.InitWeight(scale / math.sqrt(ind))
        else:
            self.InitWeight(scale / math.sqrt(outd))

    def InitWeight(self, k):
        init.uniform_(self.w.param.data, -k, k)
        init.uniform_(self.b.param.data, -k, k)
        self.init_k = k
       
    def forward(self, x):
        # 用到了floatparam和parm相加合并为真正的权重
        w = self.w() # bsz * in * out
        b = self.b().unsqueeze(1) # bsz * out
        x = x.unsqueeze(1) # bsz * 1 * in
        return torch.baddbmm(b, x, w).squeeze(1)

class HebbLinear(PlasticLinear):

    def __init__(self, ind, outd, scale=1, fan_in=False, activation='none'):
        super().__init__(ind, outd, scale, fan_in)
        self.non_linearity = Activation(activation)
    
    def process(self,x,out):
        self.w.pre = x
        self.w.post = out
        self.w.dw = torch.bmm(x.unsqueeze(-1), out.unsqueeze(-2))

    def forward(self, x):
        out = super().forward(x)
        out = self.non_linearity(out)
        self.process(x,out)
        return out

class STDPLinear(HebbLinear):

    def process(self, x, out):
        self.w.pre = x
        self.w.post = out

        hebb = torch.bmm(self.w.pre.unsqueeze(-1), self.w.post.unsqueeze(-2))
        if self.w.dw is None:
            self.w.dw = self.w.decay * hebb
        else:
            self.w.dw = (1 - self.w.decay) * self.w.dw + self.w.decay * hebb
    
    def reset_stdp(self):
        self.w.dw = None


class Linear(nn.Linear):

    def __init__(self, ind, outd, scale=1, fan_in=False):
        super().__init__(ind, outd)
        if fan_in:
            self.InitWeight(scale / math.sqrt(ind))
        else:
            self.InitWeight(scale / math.sqrt(outd))
        
    def InitWeight(self, k):
        init.uniform_(self.weight.data, -k, k)
        init.uniform_(self.bias.data, -k, k)
        self.init_k = k