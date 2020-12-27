from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda
from model import LapSRN
import numpy as np
import torch.optim as optim
from test_util import *
# Argument settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--input', type=str, required=False, default='73-512pix-speed7-ave1.tif', help='input image to use')
parser.add_argument('--model_r', type=str, default='LapSRN_model_epoch_100.pth', help='model file to use')
parser.add_argument('--model_g', type=str, default='LapSRN_model_epoch_100.pth', help='model file to use')
parser.add_argument('--model_b', type=str, default='LapSRN_model_epoch_100.pth', help='model file to use')
parser.add_argument('--outputHR2', type=str, default='73_LapSRN_R_epochs100_HR2.tif', help='where to save the output image')
parser.add_argument('--outputHR4', type=str, default='73_LapSRN_R_epochs100_HR4.tif', help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)

model_r = LapSRN().cuda()
model_g = LapSRN().cuda()
model_b = LapSRN().cuda()
optimizer_r = optim.Adagrad(model_r.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_g = optim.Adagrad(model_g.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_b = optim.Adagrad(model_b.parameters(), lr=1e-3, weight_decay=1e-5)


model_r, optimizer_r, epochs_r = load_model(model_r, optimizer_r, opt.model_r)
model_g, optimizer_g, epochs_g = load_model(model_g, optimizer_g, opt.model_g)
model_b, optimizer_b, epochs_b = load_model(model_b, optimizer_b, opt.model_b)



img = Image.open(opt.input).convert('RGB')
LR_r, LR_g, LR_b = img.split()

LR_r, LR_g, LR_b = pre_deal(LR_r, LR_g, LR_b)


HR_2_r, HR_4_r = testing(model_r, LR_r)
HR_2_g, HR_4_g = testing(model_g, LR_g)
HR_2_b, HR_4_b = testing(model_g, LR_b)


out_HR_2_r, out_HR_4_r = post_deal(HR_2_r, HR_4_r)
out_HR_2_g, out_HR_4_g = post_deal(HR_2_g, HR_4_g)
out_HR_2_b, out_HR_4_b = post_deal(HR_2_b, HR_4_b)

out_HR_2_r = Image.merge('RGB', [out_HR_2_r, out_HR_2_g, out_HR_2_b]).convert('RGB')
out_HR_4_r = Image.merge('RGB', [out_HR_4_r, out_HR_4_g, out_HR_4_b]).convert('RGB')

out_HR_2_r.save(opt.outputHR2)
print('output image saved to ', opt.outputHR2)

out_HR_4_r.save(opt.outputHR4)
print('output image saved to ', opt.outputHR4)

    

