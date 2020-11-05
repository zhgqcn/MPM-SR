from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop
import torchvision

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

model_r = torch.load(opt.model_r)
model_g = torch.load(opt.model_g)
model_r = model_r.cuda()
model_g = model_g.cuda()

transform = Compose(
    [
        ToTensor(),
    ])

img = Image.open(opt.input).convert('RGB')
r, g, _ = img.split()

r = transform(r)
r = r.unsqueeze(0)
r = r.cuda()

g = transform(g)
g = g.unsqueeze(0)
g = g.cuda()

HR_2_r, HR_4_r = model_r(r)
HR_2_g, HR_4_g = model_g(g)

black_2 = torch.zeros(1, 1024, 1024).unsqueeze(0).cuda()
HR_2 = torch.cat((HR_2_r.squeeze(0), HR_2_g.squeeze(0), black_2.squeeze(0))).unsqueeze(0)
black_4 = torch.zeros(1, 2048, 2048).unsqueeze(0).cuda()
HR_4 = torch.cat((HR_4_r.squeeze(0), HR_4_g.squeeze(0), black_4.squeeze(0))).unsqueeze(0)

torchvision.utils.save_image(HR_2, opt.outputHR2, padding=0)  # save the output
torchvision.utils.save_image(HR_4, opt.outputHR4, padding=0)
print("saved !")
    

