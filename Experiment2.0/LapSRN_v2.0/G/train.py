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

results = {'Avg. Loss': [], 'Avg. PSNR1': [], 'Avg. PSNR2': []}

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
model = LapSRN().to(device)
Loss = Loss()
criterion = nn.MSELoss()
if cuda:
    Loss = Loss.cuda()
    criterion = criterion.cuda()


def train(epoch):
        epoch_loss = 0
        for iteration, batch in enumerate(training_data_loader, 1):
            LR, HR_2_target, HR_4_target = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            optimizer.zero_grad()
            HR_2, HR_4 = model(LR)

            loss1 = Loss(HR_2, HR_2_target)
            loss2 = Loss(HR_4, HR_4_target)
            loss = loss1+loss2

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
        results['Avg. Loss'].append(float('%.4f'%(epoch_loss / len(training_data_loader))))


def val():
    avg_psnr1 = 0
    avg_psnr2 = 0
    with torch.no_grad():
        for batch in val_data_loader:
            LR, HR_2_target, HR_4_target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            HR_2, HR_4 = model(LR)
            mse1 = criterion(HR_2, HR_2_target)
            mse2 = criterion(HR_4, HR_4_target)
            psnr1 = 10 * log10(1 / mse1.item())
            psnr2 = 10 * log10(1 / mse2.item())
            avg_psnr1 += psnr1
            avg_psnr2 += psnr2
        print("===> Avg. PSNR1: {:.4f} dB".format(avg_psnr1 / len(val_data_loader)))
        print("===> Avg. PSNR2: {:.4f} dB".format(avg_psnr2 / len(val_data_loader)))
        results['Avg. PSNR1'].append(float('%.2f'%(avg_psnr1 / len(val_data_loader))))
        results['Avg. PSNR2'].append(float('%.2f'%(avg_psnr2 / len(val_data_loader))))


def checkpoint(epoch):
    model_out_g_path = "LapSRN_model_epoch_g_{}.pth".format(epoch)
    state_g = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'lr':lr}
    torch.save(state_g, model_out_g_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_g_path))


lr = opt.lr

for epoch in range(1, opt.nEpochs + 1):

    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-5)

    train(epoch)
    val()
    if epoch % 10 ==0:
        checkpoint(epoch)
    if epoch % 50 ==0:
        lr = lr/2

    data_frame = pd.DataFrame(
    data={ 'Avg. Loss': results['Avg. Loss'],
           'Avg. PSNR1': results['Avg. PSNR1'],
           'Avg. PSNR2': results['Avg. PSNR2']},
    index=range(1,epoch + 1))
    data_frame.to_csv('./result-g.csv', index_label='Epoch')

