import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from PIL import ImageEnhance

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def image_augument(image, brightness=2.0, contrast=2.0):  
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened = enh_bri.enhance(brightness)
    enh_con = ImageEnhance.Contrast(image_brightened)
    image_contrasted = enh_con.enhance(contrast)
    return  image_contrasted


def load_img(filepath, channel = 'r'):
    img = Image.open(filepath).convert('RGB')
    r, g, b = img.split()
    if channel == 'r':
        return r
    else:
        return g


class DatasetFromFolder(data.Dataset):
    def __init__(self, LR_dir, HR_2_dir, HR_4_dir, LR_transform=None, HR_2_transform=None, HR_4_transform=None, channel='r'):
        super(DatasetFromFolder, self).__init__()
        self.image_LRfilenames = [join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)]
        self.image_LRfilenames.sort()
        #print(self.image_LRfilenames)
        self.image_HR_2filenames = [join(HR_2_dir, x) for x in listdir(HR_2_dir) if is_image_file(x)]
        self.image_HR_2filenames.sort()
        #print(self.image_HR_2filenames)
        self.image_HR_4filenames = [join(HR_4_dir, x) for x in listdir(HR_4_dir) if is_image_file(x)]
        self.image_HR_4filenames.sort()
        #print(self.image_HR_4filenames)
        self.LR_transform = LR_transform
        self.HR_2_transform = HR_2_transform
        self.HR_4_transform = HR_4_transform
        self.channel = channel


    def __getitem__(self, index):
        LR = load_img(self.image_LRfilenames[index], self.channel)
        #print(self.image_LRfilenames[index])
        HR_2 = load_img(self.image_HR_2filenames[index], self.channel)
        HR_2 = image_augument(HR_2)
        HR_4 = load_img(self.image_HR_4filenames[index], self.channel)
        HR_4 = image_augument(HR_4)
        if self.LR_transform:
            LR = self.LR_transform(LR)
        if self.HR_2_transform:
            HR_2 = self.HR_2_transform(HR_2)
        if self.HR_4_transform:
            HR_4 = self.HR_4_transform(HR_4)
        return LR, HR_2, HR_4

    def __len__(self):
        return len(self.image_LRfilenames)
