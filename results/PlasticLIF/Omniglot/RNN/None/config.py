from configs.BaseConfig import BaseConfig, OmniglotConfig, miniImageNetConfig

class Config(OmniglotConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.cnn = 'ProtoNet'
        self.rnn = 'LIF_RNN'
        self.plasticity_mode = 'none'
        self.save_path = f'./result'

        self.batch_size = 8
        self.log_epoch = 10
        self.train_data = 2560000
        self.train_epoch = 1000
        self.train_batch = self.train_data // self.train_epoch // self.batch_size
        self.test_data = 12800
        self.test_epoch = 1
        self.test_batch = self.test_data // self.test_epoch // self.batch_size    

