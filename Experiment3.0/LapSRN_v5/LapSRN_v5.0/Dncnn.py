from data import After_Denoising_transform
import torch
import torchvision
from models.network_dncnn import DnCNN as net


# ----------------------------------------
# denoising model
# ----------------------------------------
def Denosing(denoise_model, input, device):
    
    model = net(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(denoise_model), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    out_denosing = model(input)

    out_denosing_lambda = After_Denoising_transform(out_denosing.squeeze(0))
    
    return out_denosing_lambda
    