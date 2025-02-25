from configs.BaseConfig import BaseConfig, OmniglotConfig, miniImageNetConfig

class Config(miniImageNetConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.batch_size = 8

        self.cnn = 'ResNet'
        self.rnn = 'RNN'
        self.plasticity_mode = 'stdp' 

        self.flag = "wds"

