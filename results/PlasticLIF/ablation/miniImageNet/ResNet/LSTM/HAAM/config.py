from configs.BaseConfig import BaseConfig, OmniglotConfig, miniImageNetConfig

class Config(miniImageNetConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.batch_size = 16

        self.cnn = 'ResNet'
        self.rnn = 'LIF_LSTM'
        self.plasticity_mode = 'stdp' 

        self.flag = "wds"

        self.num_workers = 0

