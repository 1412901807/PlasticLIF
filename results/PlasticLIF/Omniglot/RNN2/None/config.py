from configs.BaseConfig import BaseConfig, OmniglotConfig, miniImageNetConfig

class Config(OmniglotConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.batch_size = 16

        self.cnn = 'ProtoNet'
        self.rnn = 'LIF_RNN2'
        self.plasticity_mode = 'none'

        self.flag = "wds"

