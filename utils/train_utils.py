import os.path as osp

import torch
from torch.nn.utils import clip_grad_norm_
import models.model as models
from tasks import taskfunctions
from configs.config_global import DEVICE

def grad_clipping(model, max_norm, printing=False):
    p_req_grad = [p for p in model.parameters() if p.requires_grad]

    if printing:
        grad_before = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_before += param_norm.item() ** 2
        grad_before = grad_before ** (1. / 2)

    clip_grad_norm_(p_req_grad, max_norm)

    if printing:
        grad_after = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_after += param_norm.item() ** 2
        grad_after = grad_after ** (1. / 2)
        
        if grad_before > grad_after:
            print("clipped")
            print("before: ", grad_before)
            print("after: ", grad_after)


def model_init(config_, mode, net_num=None):
    if net_num is not None:
        assert mode == 'eval', 'net number only provided at eval mode'

    if config_.model_type == 'Plasticmodel':
        model = models.PlasticLinearmodel(config_)

    else:
        raise NotImplementedError("Model not Implemented")
        
    model.to(DEVICE)
    return model

def task_init(config_):
    """initialize tasks"""

    task_func_ = taskfunctions.FSC(config_)

    return task_func_
