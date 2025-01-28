from configs.BaseConfig import BaseConfig, OmniglotConfig, miniImageNetConfig

class Config(miniImageNetConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.cnn = 'ProtoNet'
        self.rnn = 'LIF_MLP'
        self.plasticity_mode = 'none'
