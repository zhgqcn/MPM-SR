from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda
from model import LapSRN
import numpy as np
import torch.optim as optim

# Argument settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--input', type=str, required=False, default='73-512pix-speed7-ave1.tif', help='input image to use')
parser.add_argument('--model', type=str, default='LapSRN_model_epoch_100.pth', help='model file to use')
parser.add_argument('--outputHR2', type=str, default='73_LapSRN_R_epochs100_HR2.tif', help='where to save the output image')
parser.add_argument('--outputHR4', type=str, default='73_LapSRN_R_epochs100_HR4.tif', help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)

model = LapSRN().cuda()
optimizer = optim.Adagrad(model.parameters(), lr=1e-3, weight_decay=1e-5)


checkpoint = torch.load(opt.model)
model.load_state_dict(checkpoint['model'])
model.eval()
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']


transform = Compose([ToTensor(),
                     Lambda(lambda x: x.repeat(3,1,1)),
                    ])

img = Image.open(opt.input).convert('RGB')
LR_r, _, _ = img.split()


LR_r = transform(LR_r)
LR_r = LR_r.unsqueeze(0)

if opt.cuda:
    LR_r = LR_r.cuda()
HR_2_r, HR_4_r = model(LR_r)
HR_2_r = HR_2_r.cpu()
HR_4_r = HR_4_r.cpu()


out_HR_2_r = HR_2_r.data[0].numpy() 
out_HR_2_r *= 255.0
out_HR_2_r = out_HR_2_r.clip(0, 255)
out_HR_2_r = Image.fromarray(np.uint8(out_HR_2_r[0]), mode='L')


out_HR_4_r = HR_4_r.data[0].numpy()
out_HR_4_r *= 255.0
out_HR_4_r = out_HR_4_r.clip(0, 255)
out_HR_4_r = Image.fromarray(np.uint8(out_HR_4_r[0]), mode='L')

out_HR_2_r.save(opt.outputHR2)
print('output image saved to ', opt.outputHR2)

out_HR_4_r.save(opt.outputHR4)
print('output image saved to ', opt.outputHR4)

    

