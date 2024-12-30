import  torch.utils.data as data
import pickle
import  os
import  os.path
import  errno
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np
import torch
import time

class Omniglot(data.Dataset):
    def __init__(self, root_path, image_size = 105):
        self.root = root_path
        self.image_size = image_size

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.data = []
        self.label = []

        self.get_dataset()

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'images_evaluation')) and os.path.exists(os.path.join(self.root, 'images_background'))
    
    def resize_image(self, image):
        if image.mode == '1':  # 如果图像是二值图像
            image = image.convert('L')  # 转换为灰度图像
        
        image = np.array(image)  # 将图像转换为 NumPy 数组
        return image

    def get_dataset(self):
        all_items = find_classes(self.root)
        idx_classes = index_classes(all_items)
        data_all = []
        label_all = []

        for index in range(len(all_items)):
            filename = all_items[index][0]
            img_path = os.path.join(all_items[index][2], filename)
            image = Image.open(img_path)
            image = self.resize_image(image)  # 调整图像大小
            image_label = idx_classes[all_items[index][1]]
            data_all.append(image)
            label_all.append(image_label)

        # 重新映射标签
        min_label_all = min(label_all)
        label_all = [x - min_label_all for x in label_all]

        # 保存不同划分的数据集
        self.save_split(data_all, label_all, 'train', 0, 1200)
        self.save_split(data_all, label_all, 'test', 1200, 1500)
        self.save_split(data_all, label_all, 'val', 1500, 1623)
    
    def save_split(self, data_all, label_all, split_name, start, end):
        self.data = []
        self.label = []
        for i in range(len(label_all)):
            if end > label_all[i] >= start:
                self.data.append(data_all[i])
                self.label.append(label_all[i])

        # 将数据转换为 NumPy 数组
        self.data = np.array(self.data)  # 将列表转换为 NumPy 数组

        with open(f'{split_name}.pickle', 'wb') as f:
            pickle.dump({'data': self.data, 'labels': self.label}, f)
        print(f'== Dataset: Found {len(self.data)} items in {split_name} set')
        

def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    # print("== Found %d items " % len(retour))
    return retour

def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    # print("== Found %d classes" % len(idx))
    return idx


Omniglot(root_path='/data/datasets/omniglot-py')