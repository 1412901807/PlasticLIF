class DatasetIters(object):

    def __init__(self, config, phase, b_size):
        
        self.data_loader = init_single_dataset(phase, b_size, config)
        self.reset()

    def reset(self):
        self.data_iter = iter(self.data_loader)
        self.iter_len = len(self.data_iter)
        # print(f"self.data_iter type: {type(self.data_iter)}") #loaderiter


def init_single_dataset(phase, b_size, config):
    num_wks = config.num_workers
    train_flag = phase == 'train'

    dataset = config.image_dataset

    if dataset == 'miniImageNet':
        from datasets.fsc.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset = MiniImageNet(phase=phase)
        data_loader = FewShotDataloader
    elif dataset == 'CIFAR_FS':
        from datasets.fsc.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset = CIFAR_FS(phase=phase)
        data_loader = FewShotDataloader
    elif dataset == 'Omniglot':
        from datasets.fsc.Omniglot import Omniglot, FewShotDataloader
        dataset = Omniglot(phase=phase)
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    
    if train_flag:
        data_loader = data_loader(
            dataset=dataset,
            nKnovel=config.train_way, # 新类别数量
            nKbase=0, # 不使用基础类别进行训练
            nExemplars=config.train_shot,  # 每个新类别的训练样本数量
            nTestNovel=config.train_query * config.train_way,  # 每个新类别的测试样本数量
            nTestBase=0, # 不使用基础类别的测试样本
            batch_size=b_size,
            num_workers=num_wks,
            epoch_size=b_size * config.train_batch, # num of data per epoch
        )
    else:
        data_loader = data_loader(
            dataset=dataset,
            nKnovel=config.train_way,
            nKbase=0,
            nExemplars=config.train_shot, 
            nTestNovel=config.train_query * config.train_way, 
            nTestBase=0, # num test examples for all the base categories
            batch_size=b_size,
            num_workers=0,
            epoch_size=b_size * config.test_batch, # num of data per epoch
        )
    data_loader = data_loader

                            
    return data_loader
