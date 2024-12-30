import os.path as osp

import torch

NP_SEED = 1234
TCH_SEED = 2147483647
TCHCUDA_SEED = 2147483647

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")