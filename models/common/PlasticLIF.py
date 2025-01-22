from braincog.base.node import *
from models.plastic.meta import PlasticParam
import torch.nn.init as init

class PlasticLIF(LIFNode):
    def __init__(self, threshold=0.5, tau=2., act_fun=QGateGrad, ind=1, scale=1, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        
        self.scale = scale
        self.ind = ind
        
        # 初始化权重
        self.w = PlasticParam(torch.empty(ind, ind))
        self.InitWeight(scale / math.sqrt(ind))

    def InitWeight(self, k):
        init.uniform_(self.w.param.data, -k, k)
        self.init_k = k

    # 修改
    def integral(self, inputs):
        w = self.w() 
        tmp = (inputs - self.mem).unsqueeze(1)
        self.mem = self.mem + torch.baddbmm(torch.zeros_like(tmp), tmp, w).squeeze(1) / self.tau

    def n_reset(self):
        """
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
