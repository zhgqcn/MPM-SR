from __future__ import print_function
import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LapSRN_r, LapSRN_g
from data import get_training_set, get_val_set
from loss import Loss
import pandas as pd
from os.path import join
import os

results = {'Avg. Loss_r': [], 'Avg. Loss_g': [],'Avg. PSNR1': [], 'Avg. PSNR2': []}

# Training settings 
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='val batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=1e-3')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, default="/content/drive/My Drive/app/DataSet/", help='root of DataSet')
parser.add_argument('--save_train_csv', type=str, default='./train.csv',help='the root of saving the train results')
parser.add_argument('--save_val_csv', type=str, default='/val.csv', help='the root of saving the val results')
parser.add_argument('--save_models', type=str, default='./', help='the root of saving the models')
parser.add_argument('--pre_model_r', type=str, default='./r.pt', help='continue train model : load r model')
parser.add_argument('--pre_model_g', type=str, default='./g.pt', help='continue train model : load g model')
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


print('===> Loading pre_train model and Building model')
model_r = LapSRN_r().to(device)
model_g = LapSRN_g().to(device)
Loss = Loss()
criterion = nn.MSELoss()
if cuda:
    Loss = Loss.cuda()
    criterion = criterion.cuda()


def train(epoch):
        epoch_loss_r,epoch_loss_g = 0, 0
        for iteration, batch in enumerate(training_data_loader, 1):
            LR_r, LR_g, HR_2_target, HR_4_target, HR_2_r_lambda, HR_2_g_lambda, HR_4_r_lambda, HR_4_g_lambda = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device), batch[6].to(device), batch[7].to(device)

            optimizer_r.zero_grad()
            optimizer_g.zero_grad()

            HR_2_r, HR_4_r = model_r(LR_r)
            HR_2_g, HR_4_g = model_g(LR_g)


            black_HR_2 = torch.zeros(3, HR_2_target[0].shape[1], HR_2_target[0].shape[2]).unsqueeze(0).to(device)
            HR_2 = torch.cat((HR_2_r[0][0].unsqueeze(0), HR_2_g[0][0].unsqueeze(0), black_HR_2[0][0].unsqueeze(0))).unsqueeze(0)
            
            black_HR_4 = torch.zeros(3, HR_4_target[0].shape[1], HR_4_target[0].shape[2]).unsqueeze(0).to(device)
            HR_4 = torch.cat((HR_4_r[0][0].unsqueeze(0), HR_4_g[0][0].unsqueeze(0), black_HR_4[0][0].unsqueeze(0))).unsqueeze(0)

            loss1_all = Loss(HR_2, HR_2_target)
            loss2_all = Loss(HR_4, HR_4_target)

            loss1_r = Loss(HR_2_r, HR_2_r_lambda)
            loss2_r = Loss(HR_4_r, HR_4_r_lambda)

            loss1_g = Loss(HR_2_g, HR_2_g_lambda)
            loss2_g = Loss(HR_4_g, HR_4_g_lambda)

            loss_r = loss1_all + loss2_all + loss1_r + loss2_r 
            loss_g = loss1_all + loss2_all + loss1_g + loss2_g

            epoch_loss_r += loss_r.item()
            epoch_loss_g += loss_g.item()
            loss_r.backward(retain_graph=True)
            loss_g.backward()
            optimizer_r.step()
            optimizer_g.step()

        print("===> Epoch {} Complete: Avg. Loss_r: {:.4f}".format(epoch, epoch_loss_r / len(training_data_loader)))
        print("===> Epoch {} Complete: Avg. Loss_g: {:.4f}".format(epoch, epoch_loss_g / len(training_data_loader)))
        results['Avg. Loss_r'].append(float('%.4f'%(epoch_loss_r / len(training_data_loader))))
        results['Avg. Loss_g'].append(float('%.4f'%(epoch_loss_g / len(training_data_loader))))


def val():
    avg_psnr1 = 0
    avg_psnr2 = 0
    with torch.no_grad():
        for batch in val_data_loader:
            
            LR_r, LR_g, HR_2_target, HR_4_target, HR_2_r_lambda, HR_2_g_lambda, HR_4_r_lambda, HR_4_g_lambda = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device), batch[6].to(device), batch[7].to(device)
            
            HR_2_r, HR_4_r = model_r(LR_r)
            HR_2_g, HR_4_g = model_r(LR_g)

            black_HR_2 = torch.zeros(3, HR_2_target[0].shape[1], HR_2_target[0].shape[2]).unsqueeze(0).to(device)
            HR_2 = torch.cat((HR_2_r[0][0].unsqueeze(0), HR_2_g[0][0].unsqueeze(0), black_HR_2[0][0].unsqueeze(0))).unsqueeze(0)
            
            black_HR_4 = torch.zeros(3, HR_4_target[0].shape[1], HR_4_target[0].shape[2]).unsqueeze(0).to(device)
            HR_4 = torch.cat((HR_4_r[0][0].unsqueeze(0), HR_4_g[0][0].unsqueeze(0), black_HR_4[0][0].unsqueeze(0))).unsqueeze(0)

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
    model_out_r_path = join(opt.save_models, 'SiameseLapSRN_r_epoch_{}.pt'.format(epoch))
    model_out_g_path = join(opt.save_models, 'SiameseLapSRN_g_epoch_{}.pt'.format(epoch))
    state_r = {'model': model_r.state_dict(), 'optimizer': optimizer_r.state_dict(), 'epoch':epoch, 'lr':lr_r}
    state_g = {'model': model_g.state_dict(), 'optimizer': optimizer_g.state_dict(), 'epoch':epoch, 'lr':lr_g}
    torch.save(state_r, model_out_r_path, _use_new_zipfile_serialization=False)
    torch.save(state_g, model_out_g_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to:\n {} \n {}".format(model_out_r_path, model_out_g_path))


if os.path.exists(opt.pre_model_r) and os.path.exists(opt.pre_model_g):
    model_r = LapSRN_r().cuda()
    model_g = LapSRN_g().cuda()

    checkpoint_r = torch.load(opt.pre_model_r)
    model_r.load_state_dict(checkpoint_r['model'])
    model_r.train()
    epochs_r = checkpoint_r['epoch']
    

    checkpoint_g = torch.load(opt.pre_model_g)
    model_g.load_state_dict(checkpoint_g['model'])
    model_g.train()
    epochs_g = checkpoint_g['epoch']
    
    optimizer_r = optim.Adagrad(model_r.parameters())
    optimizer_g = optim.Adagrad(model_g.parameters())

    for epoch in range(epochs_g + 1, opt.nEpochs + 1):

        if epoch == epochs_g + 1:
            optimizer_r.load_state_dict(checkpoint_r['optimizer'])
            optimizer_g.load_state_dict(checkpoint_g['optimizer'])
            lr_r = checkpoint_r['lr']
            lr_g = checkpoint_g['lr']
        else:
            optimizer_r = optim.Adagrad(model_r.parameters(), lr = lr_r, weight_decay = 1e-5)
            optimizer_g = optim.Adagrad(model_g.parameters(), lr = lr_g, weight_decay = 1e-5)
            
            
        train(epoch)
        val()

        if epoch % 10 ==0:
            checkpoint(epoch)
        if epoch % 50 ==0:
            lr_r = lr_r / 2
            lr_g = lr_g / 2

        data_frame = pd.DataFrame(
        data={ 'Avg. Loss_r': results['Avg. Loss_r'],
               'Avg. Loss_g': results['Avg. Loss_g'],
               'Avg. PSNR1': results['Avg. PSNR1'],
               'Avg. PSNR2': results['Avg. PSNR2']},
        index=range(epochs_g + 1, epoch + 1))
        data_frame.to_csv('./result_continue.csv', index_label='Epoch')

