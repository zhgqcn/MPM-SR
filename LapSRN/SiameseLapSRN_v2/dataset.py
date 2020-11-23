import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def load_img(filepath, type):
    img = Image.open(filepath).convert('RGB')
    if type == 'LR':
        r, g, b = img.split()
        return r, g, b
    else:
        return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, LR_dir, HR_2_dir, HR_4_dir, LR_transform = None, HR_2_transform = None, HR_4_transform = None):
        super(DatasetFromFolder, self).__init__()
        self.image_LRfilenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.image_LRfilenames.sort()
        # print(self.image_LRfilenames)
        self.image_HR_2filenames = [join(HR_2_dir, x) for x in listdir(HR_2_dir) if is_image_file(x)]
        self.image_HR_2filenames.sort()
        # print(self.image_HR_2filenames)
        self.image_HR_4filenames = [join(HR_4_dir, x) for x in listdir(HR_4_dir) if is_image_file(x)]
        self.image_HR_4filenames.sort()
        # print(self.image_HR_4filenames)
        self.LR_transform = LR_transform
        self.HR_2_transform = HR_2_transform
        self.HR_4_transform = HR_4_transform

    def __getitem__(self, index):
        LR_r, LR_g, LR_b = load_img(self.image_LRfilenames[index], 'LR')
        # print(self.image_LRfilenames[index])
        HR_2 = load_img(self.image_HR_2filenames[index], 'HR_2')
        # print(self.image_HR_2filenames[index])
        HR_4 = load_img(self.image_HR_4filenames[index], 'HR_4')
        # print(self.image_HR_4filenames[index])
        if self.LR_transform:
            LR_r = self.LR_transform(LR_r)
            LR_g = self.LR_transform(LR_g)
        if self.HR_2_transform:
            HR_2 = self.HR_2_transform(HR_2)
        if self.HR_4_transform:
            HR_4 = self.HR_4_transform(HR_4)
        return LR_r, LR_g, HR_2, HR_4

    def __len__(self):
        return len(self.image_LRfilenames)
