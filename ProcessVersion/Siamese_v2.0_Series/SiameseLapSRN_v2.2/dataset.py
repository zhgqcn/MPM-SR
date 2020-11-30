import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def load_img(filepath, type):
    img = Image.open(filepath).convert('RGB')
    if type == 'LR' or type == 'HR_lambda':
        r, g, b = img.split()
        return r, g, b
    else:
        return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, LR_dir, HR_2_dir, HR_4_dir, LR_transform = None, HR_2_transform = None, 
                 HR_4_transform = None, HR_2_transform_lambda = None, HR_4_transform_lambda = None):
        super(DatasetFromFolder, self).__init__()
        self.image_LRfilenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.image_LRfilenames.sort()
        self.image_HR_2filenames = [join(HR_2_dir, x) for x in listdir(HR_2_dir) if is_image_file(x)]
        self.image_HR_2filenames.sort()
        self.image_HR_4filenames = [join(HR_4_dir, x) for x in listdir(HR_4_dir) if is_image_file(x)]
        self.image_HR_4filenames.sort()
        self.LR_transform = LR_transform
        self.HR_2_transform = HR_2_transform
        self.HR_4_transform = HR_4_transform
        self.HR_2_transform_lambda = HR_2_transform_lambda
        self.HR_4_transform_lambda = HR_4_transform_lambda

    def __getitem__(self, index):
        LR_r, LR_g, _ = load_img(self.image_LRfilenames[index], 'LR')
        HR_2 = load_img(self.image_HR_2filenames[index], 'HR_2')
        HR_4 = load_img(self.image_HR_4filenames[index], 'HR_4')
        HR_2_r, HR_2_g, _ = load_img(self.image_HR_2filenames[index], 'HR_lambda')
        HR_4_r, HR_4_g, _ = load_img(self.image_HR_4filenames[index], 'HR_lambda') 
               
        if self.LR_transform:
            LR_r = self.LR_transform(LR_r)
            LR_g = self.LR_transform(LR_g)
        if self.HR_2_transform:
            HR_2 = self.HR_2_transform(HR_2)
        if self.HR_4_transform:
            HR_4 = self.HR_4_transform(HR_4)
        if self.HR_2_transform_lambda:
            HR_2_r = self.HR_2_transform_lambda(HR_2_r)
            HR_2_g = self.HR_2_transform_lambda(HR_2_g)
        if self.HR_4_transform_lambda:
            HR_4_r = self.HR_4_transform_lambda(HR_4_r)
            HR_4_g = self.HR_4_transform_lambda(HR_4_g)
        return LR_r, LR_g, HR_2, HR_4, HR_2_r, HR_2_g, HR_4_r, HR_4_g

    def __len__(self):
        return len(self.image_LRfilenames)
