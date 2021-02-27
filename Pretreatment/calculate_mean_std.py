'''
@author: Gqq
@date: 2020/12/01
@use: for calculate the mean and std about RGB images
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda


def Dataset_transform():
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def get_training_set(root):
        
    return DatasetFromFolder(root,
                             transform=Dataset_transform(),
                             )


class DatasetFromFolder(data.Dataset):
    def __init__(self, root, transform = None):
        super(DatasetFromFolder, self).__init__()
        self.image_LRfilenames = [join(root, x) for x in listdir(root) if is_image_file(x)]      
        self.transform = transform

    def __getitem__(self, index):
        imgs = load_img(self.image_LRfilenames[index])

        if self.transform:
            imgs = self.transform(imgs)

        return imgs

    def __len__(self):
        return len(self.image_LRfilenames)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()  # 取所有数据的相同通道计算
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


train_set = get_training_set(r'D:\\GraduationProjectBackUp\\DataSet_less\\train_LR\\')
mean ,std = get_mean_and_std(train_set)
print('LR:')
print(mean, std)

train_set = get_training_set(r'D:\\GraduationProjectBackUp\\DataSet_less\\train_HR_2\\')
mean ,std = get_mean_and_std(train_set)
print('HR_2:')
print(mean, std)

train_set = get_training_set(r'D:\\GraduationProjectBackUp\\DataSet_less\\train_HR_4\\')
mean ,std = get_mean_and_std(train_set)
print('HR_4:')
print(mean, std)