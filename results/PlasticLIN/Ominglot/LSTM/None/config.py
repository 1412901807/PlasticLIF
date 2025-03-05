from configs.BaseConfig import BaseConfig, OmniglotConfig, miniImageNetConfig

class Config(OmniglotConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.batch_size = 8

        self.cnn = 'ResNet'
        self.rnn = 'LSTM'
        self.plasticity_mode = 'none' 

        self.flag = "wds"

        self.lr = self.lr / 10
