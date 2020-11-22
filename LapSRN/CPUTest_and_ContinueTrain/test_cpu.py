from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop
import torchvision
from model import LapSRN_r, LapSRN_g
import torch.optim as optim
# Argument settingsss
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--input', type=str, required=False, default='/content/drive/My Drive/TestImg/val/44-512pix-speed7-ave1.tif', help='input image to use')
parser.add_argument('--model_r', type=str, default='LapSRN_r_epoch_10.pth', help='model file to use')
parser.add_argument('--model_g', type=str, default='LapSRN_g_epoch_10.pth', help='model file to use')
parser.add_argument('--outputHR2', type=str, default='./73_LapSRN_R_epochs100_HR2.tif', help='where to save the output image')
parser.add_argument('--outputHR4', type=str, default='./73_LapSRN_R_epochs100_HR4.tif', help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)


model_rt = LapSRN_r()
model_gt = LapSRN_g()
optimizer_rt = optim.Adagrad(model_rt.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer_gt = optim.Adagrad(model_gt.parameters(), lr=1e-3, weight_decay=1e-5)

checkpoint_r = torch.load(opt.model_r,map_location='cpu')
model_rt.load_state_dict(checkpoint_r['model'])
model_rt.eval()
optimizer_rt.load_state_dict(checkpoint_r['optimizer'])
epochs_rt = checkpoint_r['epoch']

checkpoint_g = torch.load(opt.model_g,map_location='cpu')
model_gt.load_state_dict(checkpoint_g['model'])
model_rt.eval()
optimizer_gt.load_state_dict(checkpoint_g['optimizer'])
epochs_gt = checkpoint_g['epoch']


transform = Compose(
    [
        ToTensor(),
    ])

img = Image.open(opt.input).convert('RGB')
r, g, _ = img.split()

r = transform(r)
r = r.unsqueeze(0)


g = transform(g)
g = g.unsqueeze(0)


HR_2_r, HR_4_r = model_rt(r)
HR_2_g, HR_4_g = model_gt(g)

black_2 = torch.zeros(1, 128*2, 128*2).unsqueeze(0)
HR_2 = torch.cat((HR_2_r.squeeze(0), HR_2_g.squeeze(0), black_2.squeeze(0))).unsqueeze(0)
black_4 = torch.zeros(1, 128*4, 128*4).unsqueeze(0)
HR_4 = torch.cat((HR_4_r.squeeze(0), HR_4_g.squeeze(0), black_4.squeeze(0))).unsqueeze(0)

torchvision.utils.save_image(HR_2, opt.outputHR2, padding=0)  # save the output
torchvision.utils.save_image(HR_4, opt.outputHR4, padding=0)
print("saved !")
    

