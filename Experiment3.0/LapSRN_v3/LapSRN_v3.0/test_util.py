from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda
from model import LapSRN
import numpy as np
import torch.optim as optim


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

def pre_deal(LR_r, LR_g, LR_b):
    LR_r = transform(LR_r)
    LR_r = LR_r.unsqueeze(0)
    LR_g = transform(LR_g)
    LR_g = LR_g.unsqueeze(0)
    LR_b = transform(LR_b)
    LR_b = LR_b.unsqueeze(0)
    LR_r = LR_r.cuda()
    LR_g = LR_g.cuda()
    LR_b = LR_b.cuda()
    return LR_r, LR_g, LR_b

    

