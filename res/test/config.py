from configs.BaseConfig import BaseConfig

class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.image_dataset = 'Omniglot'
        self.cnn = 'ProtoNet'
        self.rnn = 'LSTM'
        self.plasticity_mode = 'hebbian'
        self.save_path = f'/home/liweiyi/PlasticLIF/res/Omniglot/Conv4/LSTM/Hebb/result'

        self.step = 4
        self.model_type = 'Plasticmodel'
        self.input_shape = [1, 105, 105]
        self.hidden_size = 256

        self.batch_size = 64
        self.log_epoch = 10
        self.train_data = 2560000
        self.train_epoch = 1000
        self.train_batch = self.train_data // self.train_epoch // self.batch_size
        self.test_data = 12800
        self.test_epoch = 1
        self.test_batch = self.test_data // self.test_epoch // self.batch_size    

