import os.path as osp
import os
import logging
import random
from datetime import datetime

import argparse
import importlib.util
import shutil
import sys

import numpy as np
import torch
import torch.optim.lr_scheduler as lrs
import math

from configs.config_global import NP_SEED, TCH_SEED, TCHCUDA_SEED
from utils.logger import Logger
from datasets.data_sets import DatasetIters
from utils.train_utils import grad_clipping, model_init, task_init

import json

def log_progress(eid, config, logger, scheduler, train_loss, log_pre_time, test_data, net, task_func, logger_func, test_loss_list, test_results_list, best_epoch):
    # 设置列宽
    col_format = {
        'Learning Rate': '{:<15.8f}',  # 宽度15，保留8位小数
        'Epoch': '{:<10d}',           # 宽度10，整数
        'BatchNum': '{:<12d}',        # 宽度12，整数
        'DataNum': '{:<12d}',         # 宽度12，整数
        'TrainLoss': '{:<12.6f}',     # 宽度12，保留6位小数
        'Time': '{:<10.6f}',          # 宽度10，保留6位小数
    }

    # 日志表头
    logger.log_tabular('Learning Rate', col_format['Learning Rate'].format(scheduler.get_lr()[0]))
    logger.log_tabular('Epoch', col_format['Epoch'].format(eid))
    logger.log_tabular('BatchNum', col_format['BatchNum'].format(eid * config.train_batch))
    logger.log_tabular('DataNum', col_format['DataNum'].format(eid * config.train_batch * config.batch_size))
    logger.log_tabular('TrainLoss', col_format['TrainLoss'].format(train_loss / config.log_epoch))
    
    # Perform evaluation
    testloss, testresult = model_eval(config, net, test_data, task_func, logger_func)
    test_loss_list.append(testloss)
    test_results_list.append(testresult)

    # Save the best model
    if testresult == max(test_results_list):
        torch.save(net.state_dict(), osp.join(config.save_path, 'net_best.pth'))
        best_epoch = eid  # 更新最佳epoch

    # Compute and log time duration
    log_post_time = datetime.now()
    log_duration = (log_post_time - log_pre_time).total_seconds()

    logger.log_tabular('Time', col_format['Time'].format(log_duration))

    # Dump logged values
    logger.dump_tabular()

    return log_post_time, test_loss_list, test_results_list, best_epoch


def model_eval(config, net, test_data, task_func, logger=None):
    col_format = {
        'TestLoss': '{:<12.6f}',      # 宽度12，保留6位小数
        'TestAcc': '{:<10.2f}',       # 宽度10，保留2位小数
    }
    with torch.no_grad():
        net.eval()

        correct = 0
        total = 0
        test_loss = 0.0

        test_data.reset()
        test_iter = test_data.data_iter
        test_len = test_data.iter_len

        # 就一个epoch，跑完完事
        for step in range(test_len):

            data = next(test_iter)
            
            result = task_func.roll(net, data, test=True)
            loss, num, num_corr = result[: 3]

            test_loss += loss
            total += num
            correct += num_corr

        test_acc = 100 * correct / total
        # 计算每个batch的平均loss
        avg_testloss = test_loss / config.test_batch

        logger.log_tabular('TestLoss', col_format['TestLoss'].format(avg_testloss))
        logger.log_tabular('TestAcc', col_format['TestAcc'].format(test_acc))

        return avg_testloss, test_acc


def model_train(config):
    np.random.seed(NP_SEED + config.seed)
    torch.cuda.manual_seed_all(TCHCUDA_SEED + config.seed)
    torch.manual_seed(TCH_SEED + config.seed)
    random.seed(config.seed)

    logging.info("training with GPU")

    # initialize network
    net = model_init(config, mode='train')

    # 打印出可塑性参数的参数量
    print(f"dim: {net.dim}")

    print(f"flag: {config.flag}")
    print(f"batch_size: {config.batch_size}")
    if config.seed != 0:
        print(f"seed: {config.seed}")
    print(f"num_workers: {config.num_workers}")
    print(f"lr: {config.lr}")

    # save config
    save_config(config, "./config.json")

    # 初始化 logger
    # 初始化 logger
    if os.path.exists(config.save_path):
        overwrite = input(f"The directory '{config.save_path}' already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() == 'y':
            # 清理文件夹
            shutil.rmtree(config.save_path)
            os.makedirs(config.save_path)
        else:
            print("Operation aborted. Exiting program.")
            exit()

    logger = Logger(output_dir=config.save_path)

    # gradient clipping
    if config.grad_clip is not None:
        logging.info("Performs grad clipping with max norm " + str(config.grad_clip))

    # initialize task
    task_func = task_init(config)

    # initialize dataset
    train_data = DatasetIters(config, 'train', config.batch_size)

    if config.perform_val:
        test_data = DatasetIters(config, 'val', config.batch_size)

    # initialize optimizer
    if config.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wdecay, amsgrad=True)
    elif config.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr,
                                    momentum=0.9, weight_decay=config.wdecay)
    else:
        raise NotImplementedError('optimizer not implemented')

    class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, last_epoch=-1):
            self.T_max = T_max
            self.eta_min = eta_min
            self.warmup_steps = warmup_steps
            super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_steps:
                return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
            else:
                cosine_step = self.last_epoch - self.warmup_steps + 1
                cosine_total = self.T_max - self.warmup_steps
                return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cosine_step / cosine_total)) / 2 
                        for base_lr in self.base_lrs]
        
    # initialize Learning rate scheduler
    if config.use_lr_scheduler:
        if config.scheduler_type == 'ExponentialLR':
            scheduler = lrs.ExponentialLR(optimizer, gamma=0.99)
        elif config.scheduler_type == 'StepLR':
            scheduler = lrs.StepLR(optimizer, 10, gamma=0.1)
        elif config.scheduler_type == 'CosineAnnealing':
            
            total_steps = config.train_epoch
            warmup_steps = total_steps // 10

            scheduler = WarmupCosineAnnealingLR(
                optimizer, 
                T_max=total_steps,
                eta_min=config.lr / 10,
                warmup_steps=warmup_steps
            )

        else:
            raise NotImplementedError('scheduler_type must be specified')


    best_epoch = 0
    test_loss_list = []
    test_results_list = []
    train_loss = 0.0
    log_pre_time = datetime.now()

    for eid in range(config.train_epoch):
        train_data.reset()
        train_len = train_data.iter_len
        train_iter = train_data.data_iter

        for bid in range(train_len):
                    
            net.train()
            
            loss = 0.0
            optimizer.zero_grad() #!梯度清零

            data = next(train_iter)
            loss += task_func.roll(net, data, train=True)
            loss.backward()

            # gradient clipping
            if config.grad_clip is not None:
                grad_clipping(net, config.grad_clip) 

            optimizer.step()
            train_loss += loss.item()

        if (eid + 1) % config.log_epoch == 0:
            log_pre_time, test_loss_list, test_results_list, best_epoch = log_progress(
                eid, config, logger, scheduler, train_loss, log_pre_time,
                test_data, net, task_func, logger, test_loss_list, test_results_list, best_epoch)
            train_loss = 0.0

        if config.use_lr_scheduler:
            scheduler.step()


    # 保存最后的模型参数
    torch.save(net.state_dict(), osp.join(config.save_path, 'net_last.pth'))
    # 使用最后的模型进行测试
    run_test(config, net, task_func, 'test_last.txt')
    
    # 使用在验证集上最佳模型进行测试
    best_model_path = osp.join(config.save_path, 'net_best.pth')
    net.load_state_dict(torch.load(best_model_path))
    run_test(config, net, task_func, 'test_best.txt')
    # 构建新的文件名，包含 best_ib
    new_model_path = osp.join(config.save_path, f'net_best_{best_epoch}.pth')
    # 重命名文件
    os.rename(best_model_path, new_model_path)

def run_test(config, net, task_func, filename='test_last.txt'):
    np.random.seed(NP_SEED)
    torch.cuda.manual_seed_all(TCHCUDA_SEED)
    torch.manual_seed(TCH_SEED)
    random.seed(0)

    test_data = DatasetIters(config, 'test', config.batch_size)
    logger = Logger(output_dir=config.save_path, output_fname=filename)
    model_eval(config, net, test_data, task_func, logger)

    logger.dump_tabular()

def load_config(config_path):
    """加载并返回指定路径的config.py文件中的Config类实例"""
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)

    # 假设Config类在模块中定义，创建实例
    config_instance = config_module.Config()  # 创建Config类的实例
    return config_instance

def save_config(config, output_path):
    config_dict = config.__dict__
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def compute_config(config):
    if config.image_dataset == 'miniImageNet':
        scale_factors = {
            "LIF_LSTM": {"none": 1.5},
            "LIF_MLP": {"stdp": 1.5, "hebbian": 1.5, "none": 2.5},
            "LIF_RNN2": {"stdp": 1.5, "hebbian": 1.5, "none": 2.5},
            "LSTM": {"none": 1.5},
            "RNN": {"stdp": 2, "hebbian": 2, "none": 3},
            "MLP": {"stdp": 2, "hebbian": 2, "none": 3},
        }
    elif config.image_dataset == 'Omniglot':
        scale_factors = {
            "LIF_LSTM": {"none": 1.5},
            "LIF_MLP": {"stdp": 1.5, "hebbian": 1.5, "none": 2.5},
            "LIF_RNN2": {"stdp": 1.5, "hebbian": 1.5, "none": 2.5},
            "LSTM": {"none": 1.5},
            "RNN": {"stdp": 2, "hebbian": 2, "none": 3},
            "MLP": {"stdp": 2, "hebbian": 2, "none": 3},
        }

    rnn_type = config.rnn
    plasticity = config.plasticity_mode
    factor = scale_factors.get(rnn_type, {}).get(plasticity, 1)

    # 更新config中需要计算修改的部分参数
    config.hidden_size = int(config.hidden_size * factor)
    config.train_batch = config.train_data // config.train_epoch // config.batch_size
    config.test_batch = config.test_data // config.test_epoch // config.batch_size   

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with a custom config file.")
    parser.add_argument('--config_path', type=str, help="Path to the config.py file.")

    args = parser.parse_args()

    # 加载指定路径的配置文件
    config = load_config(args.config_path)
    config = compute_config(config) # bug

    # 调用model_train函数，并传递配置文件中的内容
    model_train(config)

