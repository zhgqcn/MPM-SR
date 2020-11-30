from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda
import torchvision
import numpy as np
from dataset import load_img
from model import LapSRN_r, LapSRN_g
import torch.optim as optim
# Argument settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--input', type=str, required=False, default='/content/drive/My Drive/TestImg/val/44-512pix-speed7-ave1.tif', help='input image to use')
parser.add_argument('--model_r', type=str, default='LapSRN_r_epoch_10.pth', help='model file to use')
parser.add_argument('--model_g', type=str, default='LapSRN_g_epoch_10.pth', help='model file to use')
parser.add_argument('--outputHR2', type=str, default='73_LapSRN_R_epochs100_HR2.tif', help='where to save the output image')
parser.add_argument('--outputHR4', type=str, default='73_LapSRN_R_epochs100_HR4.tif', help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)

model_rt = LapSRN_r().cuda()
model_gt = LapSRN_g().cuda()
optimizer_rt = optim.Adagrad(model_rt.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_gt = optim.Adagrad(model_gt.parameters(), lr=1e-3, weight_decay=1e-5)

checkpoint_r = torch.load(opt.model_r)
model_rt.load_state_dict(checkpoint_r['model'])
model_rt.eval()
optimizer_rt.load_state_dict(checkpoint_r['optimizer'])
epochs_rt = checkpoint_r['epoch']

checkpoint_g = torch.load(opt.model_g)
model_gt.load_state_dict(checkpoint_g['model'])
model_rt.eval()
optimizer_gt.load_state_dict(checkpoint_g['optimizer'])
epochs_gt = checkpoint_g['epoch']

transform = Compose(
    [
        ToTensor(),
        Lambda(lambda x: x.repeat(3,1,1)),
    ])

img = Image.open(opt.input).convert('RGB')
r, g, _ = img.split()

r = transform(r)
r = r.unsqueeze(0)
r = r.cuda()

g = transform(g)
g = g.unsqueeze(0)
g = g.cuda()

HR_2_r, HR_4_r = model_rt(r)
HR_2_g, HR_4_g = model_gt(g)

HR_2_r = HR_2_r.cpu()
HR_2_g = HR_2_g.cpu()
HR_4_r = HR_4_r.cpu()
HR_4_g = HR_4_g.cpu()

out_HR_2_r = HR_2_r.data[0].numpy()
out_HR_2_r *= 255.0
out_HR_2_r = out_HR_2_r.clip(0, 255)
out_HR_2_r = Image.fromarray(np.uint8(out_HR_2_r[0]), mode='L')

out_HR_2_g = HR_2_g.data[0].numpy()
out_HR_2_g *= 255.0
out_HR_2_g = out_HR_2_g.clip(0, 255)
out_HR_2_g = Image.fromarray(np.uint8(out_HR_2_g[0]), mode='L')

out_HR_4_r = HR_4_r.data[0].numpy()
out_HR_4_r *= 255.0
out_HR_4_r = out_HR_4_r.clip(0, 255)
out_HR_4_r = Image.fromarray(np.uint8(out_HR_4_r[0]), mode='L')

out_HR_4_g = HR_4_g.data[0].numpy()
out_HR_4_g *= 255.0
out_HR_4_g = out_HR_4_g.clip(0, 255)
out_HR_4_g = Image.fromarray(np.uint8(out_HR_4_g[0]), mode='L')

out_HR_2_b = Image.open('/content/drive/My Drive/TestImg/all_black_1024.tif')      
out_HR_4_b = Image.open('/content/drive/My Drive/TestImg/all_black_2048.tif')      

out_img_2 = Image.merge('RGB', [out_HR_2_r, out_HR_2_b, out_HR_2_b]).convert('RGB')   
out_img_2.save(opt.outputHR2)   
    
out_img_4 = Image.merge('RGB', [out_HR_4_r, out_HR_4_b, out_HR_4_b]).convert('RGB')   
out_img_4.save(opt.outputHR4)     

print('saved in ' + opt.outputHR2 + ' and ' + opt.outputHR4)