__all__ = [
    "LIF_LSTMCell",
    "LIF_HebbianLSTMCell",
    "LIF_STDPLSTMCell",
    "LIF_RNNCell",
    "LIF_HebbianRNNCell",
    "LIF_STDPRNNCell",
    "LIF_MLPCell",
    "LIF_HebbianMLPCell",
    "LIF_STDPMLPCell",
    "LIF_RNN2Cell",
    "LIF_HebbianRNN2Cell",
    "LIF_STDPRNN2Cell",
]

from models.common.PlasticLIF import PlasticLIF
from .linear import *
from .activation import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from braincog.base.node import *

class LIF_RNN2Cell(nn.Module):
    def __init__(self, ind, outd, step):
        super().__init__()
        self.i_fc = Linear(ind, outd)
        self.h_fc = Linear(outd, outd)
        self.LIF = nn.ModuleList(LIFNode(threshold=0.3) for _ in range(2))

    def calc_dw(self, fc, pre, post):
        pass

    def reset_LIF(self):
        for lif in self.LIF:
            lif.n_reset()

    def forward(self, x: torch.Tensor, hx: torch.Tensor):
        
        pre1 = self.i_fc(x)
        post1 = self.LIF[0](pre1)
        self.calc_dw(self.LIF[0], pre1, post1)

        pre2 = self.h_fc(hx)
        post2 = self.LIF[1](pre2)
        self.calc_dw(self.LIF[1], pre2, post2)

        post = post1 + post2

        return post
    
class LIF_HebbianRNN2Cell(LIF_RNN2Cell):
    def __init__(self, ind, outd, step):
        super().__init__(ind, outd, step)
        self.LIF = nn.ModuleList(PlasticLIF(threshold=0.3, ind=ind, scale=1) for _ in range(2))
    
    def calc_dw(self, fc, pre, post):
        fc.w.dw = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

class LIF_STDPRNN2Cell(LIF_HebbianRNN2Cell):
    
    def calc_dw(self, fc, pre, post):

        hebb = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

        if fc.w.dw is None:
            fc.w.dw = fc.w.decay * hebb
        else:
            fc.w.dw = (1 - fc.w.decay) * fc.w.dw + fc.w.decay * hebb

    def reset_stdp(self):
        for lif in self.LIF:
            lif.w.dw = None

class LIF_MLPCell(nn.Module):

    def __init__(self, ind, outd, step) -> None:
        super().__init__()
        self.i_fc = Linear(ind, outd)
        self.h_fc = Linear(outd, outd)
        self.LIF = nn.ModuleList(LIFNode(threshold=0.3) for _ in range(2))

    def reset_LIF(self):
        for lif in self.LIF:
            lif.n_reset()

    def calc_dw(self, fc, pre, post):
        pass

    def forward(self, x: torch.Tensor, hx: torch.Tensor):
        
        pre1 = self.i_fc(x)
        post1 = self.LIF[0](pre1)
        self.calc_dw(self.LIF[0], pre1, post1)

        pre2 = self.h_fc(post1)
        post2 = self.LIF[1](pre2)
        self.calc_dw(self.LIF[1], pre2, post2)

        return post2

class LIF_HebbianMLPCell(LIF_MLPCell):

    def __init__(self, ind, outd, step):
        super().__init__(ind, outd, step)
        self.LIF = nn.ModuleList(PlasticLIF(threshold=0.3, ind=ind, scale=1) for _ in range(2))

    def calc_dw(self, fc, pre, post):
        fc.w.dw = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))


class LIF_STDPMLPCell(LIF_HebbianMLPCell):

    def calc_dw(self, fc, pre, post):

        hebb = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

        if fc.w.dw is None:
            fc.w.dw = fc.w.decay * hebb
        else:
            fc.w.dw = (1 - fc.w.decay) * fc.w.dw + fc.w.decay * hebb

    # 清空历史权重
    def reset_stdp(self):
        for lif in self.LIF:
            lif.w.dw = None

class LIF_RNNCell(nn.Module):
    def __init__(self, ind, outd, step):
        super().__init__()
        self.i_fc = Linear(ind, outd)
        self.h_fc = Linear(outd, outd)
        self.LIF = LIFNode(threshold=0.3)

    def calc_dw(self, fc, pre, post):
        pass

    # def reset_LIF(self):
    #     self.LIF.n_reset()

    # def forward(self, x: torch.Tensor, hx: torch.Tensor):

    #     pre = self.i_fc(x) + self.h_fc(hx)
    #     post = self.LIF(pre)
    #     self.calc_dw(self.LIF, pre, post)

    #     return post
    
class LIF_HebbianRNNCell(LIF_RNNCell):
    def __init__(self, ind, outd, step):
        super().__init__(ind, outd, step)
        self.LIF = PlasticLIF(threshold=0.3, ind=ind, scale=1)
    
    def calc_dw(self, fc, pre, post):
        fc.w.dw = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

class LIF_STDPRNNCell(LIF_HebbianRNNCell):
    
    def calc_dw(self, fc, pre, post):

        hebb = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

        if fc.w.dw is None:
            fc.w.dw = fc.w.decay * hebb
        else:
            fc.w.dw = (1 - fc.w.decay) * fc.w.dw + fc.w.decay * hebb

    def reset_stdp(self):
        self.LIF.w.dw = None

class LIF_LSTMCell(nn.Module):

    def __init__(self, ind, outd, step):
        super().__init__()
        self.scale = 2
        self.h_fc = Linear(outd, outd * 4, scale=self.scale)
        self.i_fc = Linear(ind, outd * 4, scale=self.scale)
        self.LIF = nn.ModuleList(LIFNode(threshold=0.3) for _ in range(5))

    def reset_LIF(self):
        for lif in self.LIF:
            lif.n_reset()

    def calc_dw(self, fc, pre, post):
        pass

    def forward(self, x, hidden):
        
        hx, cx = hidden
        gates = self.h_fc(hx) + self.i_fc(x) # hx:[64, 256] x:[64, 256]
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)

        ingate_post = self.LIF[0](ingate)
        self.calc_dw(self.LIF[0], ingate, ingate_post)

        forgetgate_post = self.LIF[1](forgetgate)
        self.calc_dw(self.LIF[1], forgetgate, forgetgate_post)

        cellgate_post = self.LIF[2](cellgate)
        self.calc_dw(self.LIF[2], cellgate, cellgate_post)

        outgate_post = self.LIF[3](outgate)
        self.calc_dw(self.LIF[3], outgate, outgate_post)

        cy = (forgetgate_post * cx) + (ingate_post * cellgate_post)

        cy_post = self.LIF[4](cy)
        self.calc_dw(self.LIF[4], cy, cy_post)

        hy = outgate_post * cy_post
        
        return hy, cy

class LIF_HebbianLSTMCell(LIF_LSTMCell):

    def __init__(self, ind, outd, step):
        super().__init__(ind, outd, step)
        # ind scale为1, outd *4 scale为2
        self.LIF = nn.ModuleList(PlasticLIF(threshold=0.3, ind=ind, scale=1) for _ in range(5))

    def calc_dw(self, fc, pre, post):
        fc.w.dw = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

class LIF_STDPLSTMCell(LIF_HebbianLSTMCell):

    def calc_dw(self, fc, pre, post):

        hebb = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

        if fc.w.dw is None:
            fc.w.dw = fc.w.decay * hebb
        else:
            fc.w.dw = (1 - fc.w.decay) * fc.w.dw + fc.w.decay * hebb

    def reset_stdp(self):
        for lif in self.LIF:
            lif.w.dw = None
