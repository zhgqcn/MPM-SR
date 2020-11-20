from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda

import numpy as np

# Argument settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--input', type=str, required=False, default='73-512pix-speed7-ave1.tif', help='input image to use')
parser.add_argument('--model', type=str, default='LapSRN_model_epoch_100.pth', help='model file to use')
parser.add_argument('--outputHR2', type=str, default='73_LapSRN_G_epochs100_HR2.tif', help='where to save the output image')
parser.add_argument('--outputHR4', type=str, default='73_LapSRN_G_epochs100_HR4.tif', help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)


# model = torch.load(opt.model, map_location='cpu')
model = torch.load(opt.model)
if opt.cuda:
    model = model.cuda()

transform = Compose([ToTensor(),
                     Lambda(lambda x: x.repeat(3,1,1)),
                    ])
img = Image.open(opt.input).convert('RGB')
_, LR_g, _ = img.split()
LR_g = transform(LR_g)
LR_g = LR_g.unsqueeze(0)
#LR_g = Variable(ToTensor()(LR_g)).view(1, -1, LR_g.size[1], LR_g.size[0])
if opt.cuda:
    LR_g = LR_g.cuda()
HR_2_g, HR_4_g = model(LR_g)
HR_2_g = HR_2_g.cpu()
HR_4_g = HR_4_g.cpu()
out_HR_2_g = HR_2_g.data[0].numpy()
out_HR_2_g *= 255.0
out_HR_2_g = out_HR_2_g.clip(0, 255)
out_HR_2_g = Image.fromarray(np.uint8(out_HR_2_g[0]), mode='L')
out_HR_4_g = HR_4_g.data[0].numpy()
out_HR_4_g *= 255.0
out_HR_4_g = out_HR_4_g.clip(0, 255)
out_HR_4_g = Image.fromarray(np.uint8(out_HR_4_g[0]), mode='L')

out_HR_2_g.save(opt.outputHR2)
print('output image saved to ', opt.outputHR2)
out_HR_4_g.save(opt.outputHR4)
print('output image saved to ', opt.outputHR4)

    

