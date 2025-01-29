__all__ = ['PlasticLinearmodel']

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.plastic.meta import PlasticParam

from utils.model_utils import get_cnn, get_rnn, get_linear
import models

class PlasticLinearmodel(models.PlasticModule):

    def __init__(self, config, custom_input=False):

        super().__init__()

        PlasticParam.set_elementwise_lr(config.inner_lr_mode)
        PlasticParam.set_param_grad(not config.random_network)
        PlasticParam.set_param_decay(config.decay_mode,config.decay_num)
        
        self.step = config.step

        self.cnn = get_cnn(config.cnn, config)
        out_shape = self.cnn(torch.rand(config.input_shape).unsqueeze(0)).shape

        self.proj = nn.Sequential(
            get_linear("none", out_shape[1], config.hidden_size - config.extra_input_dim, fan_in=True),
            nn.ReLU()
        )

        self.rnn = get_rnn(config.rnn, config.plasticity_mode, config.hidden_size, config.hidden_size, config.step)
        self.full_outsize = config.model_outsize + config.modulation

        self.out_fc = get_linear( #TODO:out_fc改为了none
            "none",
            config.hidden_size, 
            self.full_outsize
        )

        self.rnn_type = config.rnn

        self.plasticity_mode = config.plasticity_mode
        self.dim = self.get_param_dim()
        self.hidden_size = config.hidden_size
        self.out_dim = config.model_outsize
        self.modulation = config.modulation
               
        self.lr = config.p_lr
        self.wd = config.p_wd
        self.grad_clip = config.inner_grad_clip
        self.weight_clip = config.weight_clip
        
        self.use_layernorm = config.layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm((self.hidden_size, ))

        self.flag = config.flag
        print(f"flag: {self.flag}")

    def forward(self,input, hidden,**Kwargs):
        
        # 1reset
        self.reset_LIF()

        ret = []

        for _ in range(self.step):

            if self.dim > 0: 
                self.set_floatparam(hidden[:, :self.dim]) 

            img_input, extra_input = input
            #

            tmp = self.cnn(img_input)
            embedding = self.proj(tmp)
            embedding = torch.cat((embedding, extra_input), dim=1)

            if self.rnn_type == 'LSTM' or self.rnn_type == 'LIF_LSTM':
                x, h = self.rnn(embedding, (hidden[:, self.dim: -self.hidden_size], hidden[:, -self.hidden_size: ]))
                if self.use_layernorm:
                    x = self.layernorm(x)
                h = torch.cat((x, h), dim=1)
                x = self.out_fc(x)
            elif self.rnn_type == 'RNN' or self.rnn_type == 'MLP' or self.rnn_type == 'LIF_RNN' or self.rnn_type == 'LIF_MLP':
                h = self.rnn(embedding, hidden[:, self.dim: ])
                if self.use_layernorm:
                    h = self.layernorm(h)
                x = self.out_fc(h)

            # 创建与 x 的最后一列具有相同形状的张量，并用 self.lr 和 self.wd 的值填充它们。
            lr = torch.full_like(x[:, -1], self.lr)
            wd = torch.full_like(x[:, -1], self.wd)

            if self.modulation:
                lr = lr * torch.sigmoid(x[:, -1]) * 2
                wd = wd * torch.sigmoid(x[:, -1]) * 2

            # update the plastic weights, if there are any
            if self.dim > 0:
                floatparam = self.update_floatparam(self.flag, lr, wd, self.grad_clip, mode=self.plasticity_mode)
                if self.weight_clip is not None:
                    floatparam = torch.clip(floatparam, -self.weight_clip, self.weight_clip)
                h = torch.cat([floatparam, h], dim=1)
            #
            hidden = h
            ret.append(x[:, :self.out_dim])
        #
        return sum(ret)/len(ret), hidden

    @property
    def memory_size(self): 
        return self.dim + self.hidden_size * (1 + (self.rnn_type == 'LSTM' or self.rnn_type == 'LIF_LSTM'))
    
    def reset_LIF(self):
        # 调用cnn和rnn的reset函数
        self.cnn.reset_LIF()
        self.rnn.reset_LIF()
    
    #只包含stdp的clear_trace
    def reset_stdp(self):
        if hasattr(self.rnn,"reset_stdp"):
            self.rnn.reset_stdp()