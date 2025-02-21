import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config_global import DEVICE

def data_batch_to_device(data_b, device=DEVICE):
    if type(data_b) is torch.Tensor:
        return data_b.to(device)
    elif type(data_b) is tuple or type(data_b) is list:
        return [data_batch_to_device(data, device=device) for data in data_b]
    else:
        raise NotImplementedError("input type not recognized")

class FSC:

    def __init__(self, config):

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.way = config.train_way
        self.shot = config.train_shot
        self.query = config.train_query
        self.randomize_order = config.randomize_train_order

    def roll(self, model, data_batch, train=False, test=False, evaluate=False):
        
        model.reset_stdp()

        img_support, labels_support, img_query, labels_query, _, _ = data_batch_to_device(data_batch)

        onehot_support = F.one_hot(labels_support, num_classes=self.way)

        pred_num = 0
        correct_num = 0
        task_loss = 0
        
        bsz = img_support.shape[0]
        # print(f"memory_size: {model.memory_size}")
        hidden = torch.zeros((bsz, model.memory_size), device=DEVICE)

        for idx, (img, input, label) in  enumerate(zip(img_support.unbind(1), onehot_support.unbind(1), labels_support.unbind(1))):
            out, hidden = model((img, input), hidden)

        for i, (img, label) in enumerate(zip(img_query.unbind(1), labels_query.unbind(1))):
            out, hidden = model((img, torch.zeros_like(onehot_support[:, 0])), hidden)
            task_loss += self.criterion(out, label)

            if i == 0:
                correct_num += (torch.argmax(out, dim=-1) == label).sum().item()
                pred_num += bsz


        task_loss = task_loss / img_query.shape[1]


        if train:
            return task_loss
        elif test or evaluate:
            return task_loss.item(), pred_num, correct_num
        else:
            raise NotImplementedError("Not Implemented")