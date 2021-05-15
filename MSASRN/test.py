from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda
from model import LapSrnMS as LapSRN
import numpy as np
import torch.optim as optim

# Argument settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--input', type=str, required=False, default='73-512pix-speed7-ave1.tif', help='input image to use')
parser.add_argument('--model_r', type=str, default='LapSRN_model_epoch_r-100.pth', help='model R file to use')
parser.add_argument('--model_g', type=str, default='LapSRN_model_epoch_g-100.pth', help='model G file to use')
parser.add_argument('--outputHR2', type=str, default='LapSRN_v4.0_R_epochs100_HR2.tif', help='where to save the output image')
parser.add_argument('--outputHR4', type=str, default='LapSRN_v4.0_R_epochs100_HR4.tif', help='where to save the output image')

parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)



transform = Compose([ToTensor(),
                     Lambda(lambda x: x.repeat(3,1,1)),
                    ])

def load_model(model, optimizer, pre_model):
    checkpoint = torch.load(pre_model)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = checkpoint['epoch']
    return model, optimizer, epochs

def pre_deal(LR_r, LR_g):
    LR_r = transform(LR_r)
    LR_r = LR_r.unsqueeze(0)
    LR_g = transform(LR_g)
    LR_g = LR_g.unsqueeze(0)
    LR_r = LR_r.cuda()
    LR_g = LR_g.cuda()
    return LR_r, LR_g

def testing(model, input):
    HR_2, HR_4 = model(input)
    HR_2 = HR_2.cpu()
    HR_4 = HR_4.cpu()
    return HR_2, HR_4

def post_deal(HR_2, HR_4):
    out_HR_2 = HR_2.data[0].numpy() 
    out_HR_2 *= 255.0
    out_HR_2 = out_HR_2.clip(0, 255)
    out_HR_2 = Image.fromarray(np.uint8(out_HR_2[0]), mode='L')

    out_HR_4 = HR_4.data[0].numpy() 
    out_HR_4 *= 255.0
    out_HR_4 = out_HR_4.clip(0, 255)
    out_HR_4 = Image.fromarray(np.uint8(out_HR_4[0]), mode='L')
    return out_HR_2, out_HR_4


with torch.no_grad():
    model_r = LapSRN(5,5,4).cuda()
    model_g = LapSRN(5,5,4).cuda()

    optimizer_r = optim.Adagrad(model_r.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer_g = optim.Adagrad(model_g.parameters(), lr=1e-3, weight_decay=1e-5)


    model_r, optimizer_r, epochs_r = load_model(model_r, optimizer_r, opt.model_r)
    model_g, optimizer_g, epochs_g = load_model(model_g, optimizer_g, opt.model_g)


    img = Image.open(opt.input).convert('RGB')
    LR_r, LR_g, LR_b = img.split()

    LR_r, LR_g = pre_deal(LR_r, LR_g)

    HR_2_r, HR_4_r = testing(model_r, LR_r)
    HR_2_g, HR_4_g = testing(model_g, LR_g)



    out_HR_2_r, out_HR_4_r = post_deal(HR_2_r, HR_4_r)
    out_HR_2_g, out_HR_4_g = post_deal(HR_2_g, HR_4_g)

    # out_HR_2_b = Image.open('/content/drive/My Drive/TestImg/all_black_1024.tif') 
    # out_HR_4_b = Image.open('/content/drive/My Drive/TestImg/all_black_2048.tif') 
    width = img.size[0]
    out_HR_2_b = Image.new("L", (width * 2, width * 2))
    out_HR_4_b = Image.new("L", (width * 4, width * 4))
    out_HR_2_r = Image.merge('RGB', [out_HR_2_r, out_HR_2_g, out_HR_2_b]).convert('RGB')
    out_HR_4_r = Image.merge('RGB', [out_HR_4_r, out_HR_4_g, out_HR_4_b]).convert('RGB')

    out_HR_2_r.save(opt.outputHR2)
    print('output image saved to ', opt.outputHR2)

    out_HR_4_r.save(opt.outputHR4)
    print('output image saved to ', opt.outputHR4)




