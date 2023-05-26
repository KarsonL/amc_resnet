# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:31:58 2020

@author: Lenovo
"""

'''
copy from: https://zhuanlan.zhihu.com/p/130673468

重写 pytorch 中加载数据集的类，以适应本地数据集加载。

使用原 类 的方法，需要 在线下载 ； 或者 改变 下载地址 为本地路径 也可以。

但是 不知道为什么 我的 报错了，说类的继承 super(CIFAR10,self) 出错

原因未知。

但使用重写的类后，super  没有报错， 程序成功运行了。


'''



import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CIFAR10_IMG(Dataset):

    def __init__(self, root, train=True, transform = None, target_transform=None):
        super(CIFAR10_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        train_list=[
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',
            'data_batch_4',
            'data_batch_5'
        ]
        test_list=['test_batch']

        if self.train :
            list = train_list
        else:
            list = test_list

        self.data=[]
        self.targets=[]

        for file_name in list:
            file_path = os.path.join(root, 'cifar-10-batches-py', file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)
    
