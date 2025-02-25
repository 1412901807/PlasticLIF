from configs.BaseConfig import BaseConfig, OmniglotConfig, miniImageNetConfig

class Config(OmniglotConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.batch_size = 8

        self.cnn = 'ResNet'
        self.rnn = 'MLP'
        self.plasticity_mode = 'hebbian' 

        self.flag = "wds"
