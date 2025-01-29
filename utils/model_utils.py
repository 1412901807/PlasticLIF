from copy import deepcopy
import torch
import os.path as osp
import torch.nn as nn
import models

from configs.config_global import DEVICE
from collections import OrderedDict

def get_cnn(cnn_type, config=None):

    if cnn_type == 'ProtoNet':
        print("cnn_type == 'ProtoNet'")
        if config is not None and config.input_shape[0] == 3: # For MiniImageNet and CifarFS
            network = models.SNN_ProtoNetEmbedding(3, 64, 64, config=config)
        elif config is not None and config.input_shape[0] == 1: # For Omniglot
            network = models.SNN_ProtoNetEmbedding(1, 64, 64, config=config)

    elif cnn_type == 'ResNet':
            print("cnn_type == 'ResNet'")
            if config is not None and config.input_shape[1] == 84: # For MiniImageNet
                network = models.SNN_resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, config=config)
                # network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            elif config is not None and config.input_shape[1] == 105: # For Omniglot
                network = models.SNN_resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5, config=config)

            else:  # For CifarFS
                network = models.SNN_resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2, config=config)


    network.eval()
    network = network.to('cpu')
    return network

def get_rnn(rnn_type, plastic_mode, rnn_in_size, hidden_size, step):
        
    if rnn_type == 'RNN' and plastic_mode == 'none':
        print("rnn_type == 'SRNN' and plastic_mode == 'none'")
        rnn = models.SNN_RNNCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'RNN' and plastic_mode == 'hebbian':
        print("rnn_type == 'SRNN' and plastic_mode == 'hebbian'")
        rnn = models.SNN_HebbianRNNCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'RNN' and plastic_mode == 'stdp':
        print("rnn_type == 'SRNN' and plastic_mode == 'stdp'")
        rnn = models.SNN_STDPRNNCell(rnn_in_size,hidden_size, step)

    elif rnn_type == 'LSTM' and plastic_mode == 'none':
        print("rnn_type == 'SLSTM' and plastic_mode == 'none'")
        rnn = models.SNN_LSTMCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LSTM' and plastic_mode == 'hebbian':
        print("rnn_type == 'SLSTM' and plastic_mode == 'hebbian'")
        rnn = models.SNN_HebbianLSTMCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LSTM' and plastic_mode == 'stdp':
        print("rnn_type == 'SLSTM' and plastic_mode == 'stdp'")
        rnn = models.SNN_STDPLSTMCell(rnn_in_size,hidden_size, step)

    elif rnn_type == 'MLP' and plastic_mode == 'none':
        print("rnn_type == 'SMLP' and plastic_mode == 'none'")
        rnn = models.SNN_MLPCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'MLP' and plastic_mode == 'hebbian':
        print("rnn_type == 'SMLP' and plastic_mode == 'hebbian'")
        rnn = models.SNN_HebbianMLPCell(rnn_in_size,hidden_size, step)
    elif rnn_type == 'MLP' and plastic_mode == 'stdp':
        print("rnn_type == 'SMLP' and plastic_mode == 'stdp'")
        rnn = models.SNN_STDPMLPCell(rnn_in_size,hidden_size, step)

    elif rnn_type == 'LIF_LSTM' and plastic_mode == 'none':
        print("rnn_type == 'LIF_LSTM' and plastic_mode == 'none'")
        rnn = models.LIF_LSTMCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LIF_LSTM' and plastic_mode == 'hebbian':
        print("rnn_type == 'LIF_LSTM' and plastic_mode == 'hebbian'")
        rnn = models.LIF_HebbianLSTMCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LIF_LSTM' and plastic_mode == 'stdp':
        print("rnn_type == 'LIF_LSTM' and plastic_mode == 'stdp'")
        rnn = models.LIF_STDPLSTMCell(rnn_in_size,hidden_size, step)
    
    elif rnn_type == 'LIF_RNN' and plastic_mode == 'none':
        print("rnn_type == 'LIF_RNN' and plastic_mode == 'none'")
        rnn = models.LIF_RNNCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LIF_RNN' and plastic_mode == 'hebbian':
        print("rnn_type == 'LIF_RNN' and plastic_mode == 'hebbian'")
        rnn = models.LIF_HebbianRNNCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LIF_RNN' and plastic_mode == 'stdp':
        print("rnn_type == 'LIF_RNN' and plastic_mode == 'stdp'")
        rnn = models.LIF_STDPRNNCell(rnn_in_size,hidden_size, step)
    
    elif rnn_type == 'LIF_MLP' and plastic_mode == 'none':
        print("rnn_type == 'LIF_MLP' and plastic_mode == 'none'")
        rnn = models.LIF_MLPCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LIF_MLP' and plastic_mode == 'hebbian':
        print("rnn_type == 'LIF_MLP' and plastic_mode == 'hebbian'")
        rnn = models.LIF_HebbianMLPCell(rnn_in_size, hidden_size, step)
    elif rnn_type == 'LIF_MLP' and plastic_mode == 'stdp':
        print("rnn_type == 'LIF_MLP' and plastic_mode == 'stdp'")
        rnn = models.LIF_STDPMLPCell(rnn_in_size,hidden_size, step)

    

    else:
        raise NotImplementedError('RNN not implemented')

    return rnn

def get_linear(plastic_mode, in_size, out_size, activation='none', fan_in=False):
    
    if plastic_mode == 'none':
        layer = nn.Sequential(
            models.Linear(in_size, out_size, fan_in=fan_in),
            models.Activation(activation)
        )
    elif plastic_mode == 'hebbian':
        layer = models.HebbLinear(in_size, out_size, fan_in=fan_in, activation=activation)
        
    elif plastic_mode == 'stdp':
        layer = models.STDPLinear(in_size, out_size, fan_in=fan_in, activation=activation)
    return layer