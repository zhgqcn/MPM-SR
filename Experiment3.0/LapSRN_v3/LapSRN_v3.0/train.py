from __future__ import print_function
import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import LapSRN
from data import get_training_set, get_val_set
from loss import Loss
import pandas as pd

results = {'R_Avg. Loss': [], 'R_Avg. PSNR1': [], 'R_Avg. PSNR2': [],
           'G_Avg. Loss': [], 'G_Avg. PSNR1': [], 'G_Avg. PSNR2': [],}

# Training settings 
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='val batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=1e-3')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, default="/content/drive/My Drive/app/MMP_Dataset/", help='root of DataSet')
opt = parser.parse_args()
print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.dataset)
val_set = get_val_set(opt.dataset)
training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
val_data_loader = DataLoader(dataset=val_set, batch_size=opt.valBatchSize, shuffle=False)

print('===> Building model')
model_r = LapSRN().to(device)
Loss_r = Loss()

model_g = LapSRN().to(device)
Loss_g = Loss()

criterion = nn.MSELoss()
if cuda:
    Loss_r = Loss_r.cuda()
    Loss_g = Loss_g.cuda()
    criterion = criterion.cuda()


def train(epoch):
        epoch_loss_r, epoch_loss_g = 0, 0
        for _, batch in enumerate(training_data_loader, 1):
            LR_r, LR_g, HR_2_r_Target, HR_2_g_Target, HR_4_r_Target, HR_4_g_Target = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                batch[3].to(device), batch[4].to(device), batch[5].to(device)

            optimizer_r.zero_grad()
            HR_2_r, HR_4_r = model_r(LR_r)
            loss_r_X2 = Loss_r(HR_2_r, HR_2_r_Target)
            loss_r_X4 = Loss_r(HR_4_r, HR_4_r_Target)
            loss_r = loss_r_X2 + loss_r_X4
            epoch_loss_r += loss_r.item()
            loss_r_X2.backward(retain_graph = True)
            loss_r.backward()
            optimizer_r.step()

            optimizer_g.zero_grad()
            HR_2_g, HR_4_g = model_g(LR_g)
            loss_g_X2 = Loss_g(HR_2_g, HR_2_g_Target)
            loss_g_X4 = Loss_g(HR_4_g, HR_4_g_Target)
            loss_g = loss_g_X2 + loss_g_X4
            epoch_loss_g += loss_g.item()
            loss_g_X2.backward(retain_graph = True)
            loss_g.backward()
            optimizer_g.step()

        print("===> Epoch {} Complete: R_Avg. Loss: {:.4f}".format(epoch, epoch_loss_r / len(training_data_loader)))
        print("===> Epoch {} Complete: G_Avg. Loss: {:.4f}".format(epoch, epoch_loss_g / len(training_data_loader)))
        results['R_Avg. Loss'].append(float('%.4f'%(epoch_loss_r / len(training_data_loader))))
        results['G_Avg. Loss'].append(float('%.4f'%(epoch_loss_g / len(training_data_loader))))


def val():
    avg_psnr1_r, avg_psnr1_g = 0, 0
    avg_psnr2_r, avg_psnr2_g = 0, 0
    with torch.no_grad():
        for batch in val_data_loader:
            LR_r, LR_g, HR_2_r_Target, HR_2_g_Target, HR_4_r_Target, HR_4_g_Target = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device), \
                batch[3].to(device), batch[4].to(device), batch[5].to(device)

            HR_2_r, HR_4_r = model_r(LR_r)
            mse1_r = criterion(HR_2_r, HR_2_r_Target)
            mse2_r = criterion(HR_4_r, HR_4_r_Target)
            psnr1_r = 10 * log10(1 / mse1_r.item())
            psnr2_r = 10 * log10(1 / mse2_r.item())
            avg_psnr1_r += psnr1_r
            avg_psnr2_r += psnr2_r

            HR_2_g, HR_4_g = model_r(LR_g)
            mse1_g = criterion(HR_2_g, HR_2_g_Target)
            mse2_g = criterion(HR_4_g, HR_4_g_Target)
            psnr1_g = 10 * log10(1 / mse1_g.item())
            psnr2_g = 10 * log10(1 / mse2_g.item())
            avg_psnr1_g += psnr1_g
            avg_psnr2_g += psnr2_g

        print("===> R_Avg. PSNR1: {:.4f} dB \t R_Avg. PSNR2: {:.4f} dB".format(avg_psnr1_r / len(val_data_loader), avg_psnr2_r / len(val_data_loader)))
        print("===> G_Avg. PSNR1: {:.4f} dB \t G_Avg. PSNR2: {:.4f} dB".format(avg_psnr1_g / len(val_data_loader), avg_psnr2_g / len(val_data_loader)))
        results['R_Avg. PSNR1'].append(float('%.2f'%(avg_psnr1_r / len(val_data_loader))))
        results['R_Avg. PSNR2'].append(float('%.2f'%(avg_psnr2_r / len(val_data_loader))))
        results['G_Avg. PSNR1'].append(float('%.2f'%(avg_psnr1_g / len(val_data_loader))))
        results['G_Avg. PSNR2'].append(float('%.2f'%(avg_psnr2_g / len(val_data_loader))))


def checkpoint(epoch):
    model_out_r_path = "LapSRN_model_epoch_r_{}.pth".format(epoch)
    state_r = {'model': model_r.state_dict(), 'optimizer': optimizer_r.state_dict(), 'epoch':epoch, 'lr':lr_r}
    torch.save(state_r, model_out_r_path, _use_new_zipfile_serialization=False)
    
    model_out_g_path = "LapSRN_model_epoch_g_{}.pth".format(epoch)
    state_g = {'model': model_g.state_dict(), 'optimizer': optimizer_g.state_dict(), 'epoch':epoch, 'lr':lr_g}
    torch.save(state_g, model_out_g_path, _use_new_zipfile_serialization=False)
    
    print("Checkpoint saved to {} and {}".format(model_out_r_path, model_out_g_path))


lr_r, lr_g = opt.lr, opt.lr

for epoch in range(1, opt.nEpochs + 1):

    optimizer_r = optim.Adagrad(model_r.parameters(), lr=lr_r, weight_decay=1e-5)
    optimizer_g = optim.Adagrad(model_g.parameters(), lr=lr_g, weight_decay=1e-5)

    train(epoch)
    val()
    if epoch % 10 == 0:
        checkpoint(epoch)
    if epoch % 50 == 0:
        lr_r, lr_g = lr_r / 2, lr_g / 2

    data_frame = pd.DataFrame(
    data={ 'R_Avg. Loss': results['R_Avg. Loss'],
           'R_Avg. PSNR1': results['R_Avg. PSNR1'],
           'R_Avg. PSNR2': results['R_Avg. PSNR2'],
           'G_Avg. Loss': results['G_Avg. Loss'],
           'G_Avg. PSNR1': results['G_Avg. PSNR1'],
           'G_Avg. PSNR2': results['G_Avg. PSNR2']},
    index=range(1,epoch + 1))
    data_frame.to_csv('./result.csv', index_label='Epoch')

