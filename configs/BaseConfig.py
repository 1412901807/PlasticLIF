class BaseConfig(object):
    def __init__(self):

        # 注意train_data/train_epoch/batch_size 和 test_data/test_epoch/batch_sizes要整除
        self.step = 4
        self.hidden_size = 256
        
        self.train_way = 5
        self.train_shot = 1
        self.train_query = 1
        self.randomize_train_order = True
        self.num_workers = 4
        self.lr = 0.001
        self.wdecay = 0.0005
        self.seed = 0
        self.inner_lr_mode = 'random'
        self.random_network = False
        self.decay_mode = 'random'
        self.decay_num = 0.3
        self.extra_input_dim = 5
        self.model_outsize = 5
        self.modulation = True
        self.p_lr = 0.1
        # self.p_wd = 0.1
        self.inner_grad_clip = 1
        self.layernorm = False
        self.weight_clip = None
        self.grad_clip = 5
        self.label_smoothing = 0
        self.perform_val = True
        self.perform_test = True
        self.optimizer_type = 'AdamW'
        self.use_lr_scheduler = True
        self.scheduler_type = 'CosineAnnealing'
        self.model_type = 'Plasticmodel'

class OmniglotConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.image_dataset = 'Omniglot'
        self.input_shape = [1, 105, 105]

class miniImageNetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.image_dataset = 'miniImageNet'
        self.input_shape = [3, 84, 84]

        

