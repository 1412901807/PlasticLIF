__all__ = ['PlasticModule', 'PlasticParam']

import typing as tp
import torch
import numpy as np
import torch.nn as nn

class PlasticParam(nn.Module):

	lr_mode = 'uniform'
	requires_param_grad = True

	@classmethod
	def set_elementwise_lr(cls, mode):
		if mode is not None:
			cls.lr_mode = mode

	# 设置decay可学习还是固定值
	@classmethod
	def set_param_decay(cls, mode, num=0.7):
		if mode is not None:
			cls.decay_mode = mode
			cls.decay_num = num

	@classmethod
	def set_param_grad(cls, mode=True):
		cls.requires_param_grad = mode


	def __init__(self, param: torch.Tensor):
		super().__init__()
		# 静态权重
		self.param = nn.Parameter(param, requires_grad=self.requires_param_grad)

		if self.decay_mode == 'fix':
			self.decay = self.decay_num

		elif self.decay_mode == 'random':
			self.decay = nn.Parameter(torch.rand_like(param), requires_grad=True)

		else:
			raise NotImplementedError(f"Unrecognized mode {self.decay_mode}")
		
		print("decay_mode = {}, decay_num = {}".format(self.decay_mode, self.decay_num))
		
		if self.lr_mode == 'none':
			self.lr = 1
		elif self.lr_mode == 'uniform':
			self.lr = nn.Parameter(torch.ones_like(param))
		elif self.lr_mode == 'neg_uniform':
			self.lr = nn.Parameter(torch.full_like(param, -1))
		elif self.lr_mode == 'random':
			self.lr = nn.Parameter(torch.rand_like(param) * 2 - 1)
		else:
			raise NotImplementedError(f"Unrecognized mode {self.lr_mode}")

		# 动态权重
		self.floatparam: tp.Optional[ torch.Tensor] = None
		self.pre: tp.Optional[ torch.Tensor] = None
		self.post: tp.Optional[ torch.Tensor] = None
		self.dw: tp.Optional[ torch.Tensor] = None #New
		self.total_dim = np.prod(self.param.shape)

	def forward(self) -> torch.Tensor:
		assert self.floatparam is not None, "Parameter not initialized" 
		return self.floatparam + self.param

	def clear_floatparam(self):
		self.floatparam = None

	def set_floatparam(self, data: torch.Tensor):
		self.clear_floatparam()
		# TODO: 应该要可学习，毕竟之后都会清空
		self.floatparam = data.view(-1, *self.param.shape)
		self.floatparam.requires_grad_(True)

	def __repr__(self):
		target = self.param
		return f'{self.__class__.__name__}{tuple( target.shape)}'

class PlasticModule(nn.Module):

	def allparams(self):
		paramlist = []
		for param in self.modules():
			if isinstance(param, PlasticParam):
				paramlist.append(param)
		return paramlist

	def set_floatparam(self, data):
		count = 0
		for param in self.modules():
			if isinstance(param, PlasticParam):
				d = param.total_dim
				param.set_floatparam(data[:, count: count + d])
				count += d

	def get_param_dim(self) -> int:
		count = 0
		for param in self.modules():
			# print(type(param).__name__)
			if isinstance(param, PlasticParam):
				d = param.total_dim
				count += d
		return count

	def update_floatparam(self, lr, wd, max_norm, mode='hebbian') -> torch.Tensor:
		params = self.allparams()

		eps = 1e-8

		if mode == 'hebbian':
			grads = []
			for param in params:
				if param.param.dim() == 2:
					grad = param.dw
				else:
					grad = torch.zeros_like(param.floatparam) 
				grads.append(grad)
		
		elif mode == 'stdp':
			grads = []
			for param in params:
				if param.param.dim() == 2:
					grad = param.dw
				else:
					grad = torch.zeros_like(param.floatparam)
				grads.append(grad)

		else:
			raise NotImplementedError(mode)

		# shrink the learning rate according the norm
		norm = 0
		for grad in grads:
			norm = norm + grad.square().sum(dim=tuple(range(1, grad.dim())))
		
		norm = torch.sqrt(norm + eps) #norm [64]
		lr = lr - lr * (1 - max_norm / norm) * (norm > max_norm)

		lrs = [lr, ]
		wds = [wd, ]
		# 增加两个维度
		for d in range(2):
			lrs.append(lrs[-1].unsqueeze(-1))
			wds.append(wds[-1].unsqueeze(-1))

		# 1
		new_param_list = []
		for grad, param in zip(grads, params):
			new_param = (1 - lrs[param.param.dim()]) * param.floatparam + lrs[param.param.dim()] * grad * param.lr
			new_param_list.append(new_param)
		# print("111111111111111111111111111111111111111")
		
		
		# # 2
		# new_param_list = []
		# for grad, param in zip(grads, params):
		# 	new_param = (1 - wds[param.param.dim()]) * param.floatparam + lrs[param.param.dim()] * grad * param.lr
		# 	new_param_list.append(new_param)
		# # print("22222222222222222222222222222222222222222")

		new_param = torch.cat([param.view(param.shape[0], -1) for param in new_param_list], dim=1)
		return new_param