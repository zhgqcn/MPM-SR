import os.path
import logging
from torchvision.transforms import Compose, ToTensor, Lambda
import numpy as np
from torch.autograd import Variable
from PIL import Image
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat
import torch
import torchvision


transform = Compose([ToTensor(),
                    # Lambda(lambda x: x.repeat(3,1,1)),
                    ])

transform3 = Compose([
    Lambda(lambda x: x.repeat(3,1,1)),
])

img = Image.open('/content/drive/MyDrive/KAIR/testsets/set12/test_1.tif').convert('RGB')
LR_r, _, _ = img.split()


LR_r = transform(LR_r)
LR_r_3 = transform3(LR_r)
LR_r = LR_r.unsqueeze(0).cuda()

torchvision.utils.save_image(LR_r, './LR_out.tif', padding=0)
torchvision.utils.save_image(LR_r_3, './LR_3_out.tif', padding=0)
model_path = './model_zoo/dncnn_gray_blind.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------
# load model
# ----------------------------------------

from models.network_dncnn import DnCNN as net
model = net(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
# model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for _, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

de_out = model(LR_r)

de_out_3 = transform3(de_out.squeeze(0))
torchvision.utils.save_image(de_out, './out.tif', padding=0)
torchvision.utils.save_image(de_out_3, './out_3.tif', padding=0)
print('de_out.shape' + str(de_out.size()))

print('de_out_3.shape' + str(de_out_3.size()))

    