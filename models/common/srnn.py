__all__ = [
    "SNN_RNNCell",
    "SNN_HebbianRNNCell",
    "SNN_STDPRNNCell",
    "SNN_LSTMCell",
    "SNN_HebbianLSTMCell",
    "SNN_STDPLSTMCell",
    "SNN_MLPCell",
    "SNN_HebbianMLPCell",
    "SNN_STDPMLPCell",
]

from .linear import *
from .activation import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from braincog.base.node import *

class SNN_MLPCell(nn.Module):

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

        tmp = self.LIF[0](self.i_fc(x))
        out = self.LIF[1](self.h_fc(tmp))

        self.calc_dw(self.i_fc, x, tmp)
        self.calc_dw(self.h_fc, tmp, out)

        return out

class SNN_HebbianMLPCell(SNN_MLPCell):

    def __init__(self, ind, outd, step):
        super().__init__(ind, outd, step)
        self.i_fc = PlasticLinear(ind, outd, scale=1)
        self.h_fc = PlasticLinear(outd, outd, scale=1)

    def calc_dw(self, fc, pre, post):
        fc.w.dw = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

class SNN_STDPMLPCell(SNN_HebbianMLPCell):

    def calc_dw(self, fc, pre, post):

        hebb = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

        if fc.w.dw is None:
            fc.w.dw = fc.w.decay * hebb
        else:
            fc.w.dw = (1 - fc.w.decay) * fc.w.dw + fc.w.decay * hebb

    # 清空历史权重
    def reset_stdp(self):
        self.h_fc.w.dw = None
        self.i_fc.w.dw = None

# class SNN_RNNCell(nn.Module):

#     def __init__(self, ind, outd, step):
#         super().__init__()
#         self.i_fc = Linear(ind, outd)
#         self.h_fc = Linear(outd, outd)
#         self.LIF = LIFNode(threshold=0.3)

#     def calc_dw(self, fc, pre, post):
#         pass

#     def reset_LIF(self):
#         self.LIF.n_reset()

#     def forward(self, x: torch.Tensor, hx: torch.Tensor):

#         out = self.LIF(self.i_fc(x) + self.h_fc(hx))

#         self.calc_dw(self.i_fc, x, out)
#         self.calc_dw(self.h_fc, hx, out)

#         return out

class SNN_RNNCell(nn.Module):

    def __init__(self, ind, outd, step):
        super().__init__()
        self.i_fc = Linear(ind, outd)
        self.h_fc = Linear(outd, outd)
        self.LIF = nn.ModuleList(LIFNode(threshold=0.3) for _ in range(2))
        print("RNN2")

    def calc_dw(self, fc, pre, post):
        pass

    def reset_LIF(self):
        for lif in self.LIF:
            lif.n_reset()

    def forward(self, x: torch.Tensor, hx: torch.Tensor):
        
        post1 = self.LIF[0](self.i_fc(x))
        post2 = self.LIF[1](self.h_fc(hx))

        self.calc_dw(self.i_fc, x, post1)
        self.calc_dw(self.h_fc, hx, post2)

        post = post1 + post2

        return post

class SNN_HebbianRNNCell(SNN_RNNCell):

    def __init__(self, ind, outd, step):
        super().__init__(ind, outd, step)
        self.h_fc = PlasticLinear(outd, outd, scale=1)
        self.i_fc = PlasticLinear(ind, outd, scale=1)

    def calc_dw(self, fc, pre, post):
        fc.w.dw = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

class SNN_STDPRNNCell(SNN_HebbianRNNCell):

    def calc_dw(self, fc, pre, post):

        hebb = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

        if fc.w.dw is None:
            fc.w.dw = fc.w.decay * hebb
        else:
            fc.w.dw = (1 - fc.w.decay) * fc.w.dw + fc.w.decay * hebb

    def reset_stdp(self):
        self.h_fc.w.dw = None
        self.i_fc.w.dw = None

class SNN_LSTMCell(nn.Module):

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
        gates = self.h_fc(hx) + self.i_fc(x)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)

        ingate = self.LIF[0](ingate)
        forgetgate = self.LIF[1](forgetgate)
        cellgate = self.LIF[2](cellgate)
        outgate = self.LIF[3](outgate)

        post = torch.cat(
            [ingate, forgetgate, cellgate, outgate], dim=-1
        )  # TODO: 修改为第二种方式

        self.calc_dw(self.i_fc, x, post)
        self.calc_dw(self.h_fc, hx, post)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.LIF[4](cy)

        return hy, cy

# # 更改一版LSTM
# class SNN_LSTMCell(nn.Module):

#     def __init__(self, ind, outd, step):
#         super().__init__()
#         self.scale = 2
#         self.h_fc = Linear(outd, outd * 4, scale=self.scale)
#         self.i_fc = Linear(ind, outd * 4, scale=self.scale)
#         self.LIF = nn.ModuleList(LIFNode(threshold=0.3) for _ in range(5))
#         print("LSTM2")

#     def reset_LIF(self):
#         for lif in self.LIF:
#             lif.n_reset()

#     def calc_dw(self, fc, pre, post):
#         pass

#     def forward(self, x, hidden):

#         hx, cx = hidden

#         post1 = self.i_fc(x)
#         self.calc_dw(self.i_fc, x, post1)

#         post2 = self.h_fc(hx) 
#         self.calc_dw(self.h_fc, hx, post2)

#         gates = post1 + post2

#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)

#         ingate = self.LIF[0](ingate)
#         forgetgate = self.LIF[1](forgetgate)
#         cellgate = self.LIF[2](cellgate)
#         outgate = self.LIF[3](outgate)

#         cy = (forgetgate * cx) + (ingate * cellgate)
#         hy = outgate * self.LIF[4](cy)

#         return hy, cy

class SNN_HebbianLSTMCell(SNN_LSTMCell):

    def __init__(self, ind, outd, step):
        super().__init__(ind, outd, step)
        self.h_fc = PlasticLinear(outd, outd * 4, scale=self.scale)
        self.i_fc = PlasticLinear(ind, outd * 4, scale=self.scale)

    def calc_dw(self, fc, pre, post):
        fc.w.dw = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

class SNN_STDPLSTMCell(SNN_HebbianLSTMCell):

    def calc_dw(self, fc, pre, post):

        hebb = torch.bmm(pre.unsqueeze(-1), post.unsqueeze(-2))

        if fc.w.dw is None:
            fc.w.dw = fc.w.decay * hebb
        else:
            fc.w.dw = (1 - fc.w.decay) * fc.w.dw + fc.w.decay * hebb

    def reset_stdp(self):

        self.h_fc.w.dw = None
        self.i_fc.w.dw = None
