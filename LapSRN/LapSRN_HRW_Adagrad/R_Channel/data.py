from os.path import join

from torchvision.transforms import Compose, ToTensor, Lambda

from dataset import DatasetFromFolder


def LR_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: x.repeat(3, 1, 1)),  # 图片格式是单通道,复制使其达到3通道
    ])

def HR_2_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: x.repeat(3,1,1)),
    ])

def HR_4_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: x.repeat(3,1,1)),
    ])


def get_training_set(root):
    root_dir = root
    train_LR_dir = join(root_dir, "train_LR/")
    train_HR_2_dir = join(root_dir, "train_HR_2/")
    train_HR_4_dir = join(root_dir, "train_HR_4/")

    return DatasetFromFolder(train_LR_dir,
                             train_HR_2_dir,
                             train_HR_4_dir,
                             LR_transform=LR_transform(),
                             HR_2_transform=HR_2_transform(),
                             HR_4_transform=HR_4_transform())


def get_val_set(root):
    root_dir = root
    val_LR_dir = join(root_dir, "val_LR/")
    val_HR_2_dir = join(root_dir, "val_HR_2/")
    val_HR_4_dir = join(root_dir, "val_HR_4/")

    return DatasetFromFolder(val_LR_dir,
                             val_HR_2_dir,
                             val_HR_4_dir,
                             LR_transform=LR_transform(),
                             HR_2_transform=HR_2_transform(),
                             HR_4_transform=HR_4_transform())

