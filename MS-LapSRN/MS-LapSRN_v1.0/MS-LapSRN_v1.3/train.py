from __future__ import print_function
import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
from loss import Loss
from loss import ssim
from math import log10
from model import LapSrnMS as LapSRN
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_val_set

# settings 
parser = argparse.ArgumentParser(description='PyTorch MS-LapSRN')
parser.add_argument('--option', type=str, default='train', help='train | test | train_continue')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--valBatchSize', type=int, default=1, help='val batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=1e-3')
parser.add_argument("--step", type=int, default=50, help="Sets the learning rate")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, default="./dataset", help='root of DataSet')
parser.add_argument("--weight_decay", "--wd", default=1e-5, type=float, help="weight decay, Default: 1e-4")
parser.add_argument('--channel', type=str, default='r', help='which channel to train r | g | b')
# continue training setting
parser.add_argument('--resume', type=str, default='', help='Path to checkpoint (default: none)')
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")


def train(training_data_loader, model, optimizer, Loss, epoch):
    # training_data_loader : get the training dataset
    # model : the training model
    # optimizer, Loss, epoch : the important training parameters
    lr = adjust_learning_rate(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    epoch_loss = 0
    for _, batch in enumerate(training_data_loader, 1):
        LR, HR_2_target, HR_4_target = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        optimizer.zero_grad()

        SR_2, SR_4 = model(LR)
        loss1 = Loss(SR_2, HR_2_target)
        loss2 = Loss(SR_4, HR_4_target)
        loss = loss1 + loss2

        epoch_loss += loss.item()

        # backward by two steps
        loss1.backward(retain_graph = True)
        loss.backward()
        
        optimizer.step()

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    results['Avg. Loss'].append(float('%.4f'%(epoch_loss / len(training_data_loader))))


def val(val_data_loader, model, criterion):
    avg_psnr1, avg_psnr2 = 0, 0
    avg_ssim1, avg_ssim2 = 0, 0
    with torch.no_grad():
        for batch in val_data_loader:
            LR, HR_2_target, HR_4_target = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            SR_2, SR_4 = model(LR)
            mse1 = criterion(SR_2, HR_2_target)
            mse2 = criterion(SR_4, HR_4_target)
            psnr1 = 10 * log10(1 / mse1.item())
            psnr2 = 10 * log10(1 / mse2.item())
            ssim1 = ssim(SR_2, HR_2_target)
            ssim2 = ssim(SR_4, HR_4_target)
            avg_psnr1 += psnr1
            avg_psnr2 += psnr2
            avg_ssim1 += ssim1
            avg_ssim2 += ssim2
        print("====> Avg. PSNR1: {:.4f}     Avg. SSIM1: {:.4f} ".format(avg_psnr1 / len(val_data_loader), avg_ssim1 / len(val_data_loader)))
        print("====> Avg. PSNR2: {:.4f}     Avg. SSIM2: {:.4f} ".format(avg_psnr2 / len(val_data_loader), avg_ssim2 / len(val_data_loader)))
        results['Avg. PSNR1'].append(float('%.2f'%(avg_psnr1 / len(val_data_loader))))
        results['Avg. PSNR2'].append(float('%.2f'%(avg_psnr2 / len(val_data_loader))))
        results['Avg. SSIM1'].append(float('%.2f'%(avg_ssim1 / len(val_data_loader))))
        results['Avg. SSIM2'].append(float('%.2f'%(avg_ssim2 / len(val_data_loader))))


def checkpoint(model, optimizer, epoch, lr, channel = 'r'):
    # model : the model which want to save
    # optimizer, epoch, lr : the training parameters to save
    # channel : the train channel r or g
    model_out_path = "MSLapSRN_model_epoch_{}_{}.pth".format(channel, epoch)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'lr':lr}
    torch.save(state, model_out_path, _use_new_zipfile_serialization=False)
    print("Checkpoint saved to {}".format(model_out_path))

def adjust_learning_rate(epoch):
    # lr / 2 by epoch increase 50
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


def main():
    global opt, device, results
    results = {'Avg. Loss': [], 'Avg. PSNR1': [], 'Avg. PSNR2': [], 'Avg. SSIM1':[], 'Avg. SSIM2':[]} # save the result
    opt = parser.parse_args()
    print(opt)
    lr = opt.lr
    device = torch.device("cuda" if opt.cuda else "cpu")
    
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = get_training_set(opt.dataset, opt.channel)
    val_set = get_val_set(opt.dataset, opt.channel)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.valBatchSize, shuffle=False)


    print('===> Building model')
    model = LapSRN(5, 5, 4).to(device)
    loss = Loss()
    criterion = nn.MSELoss()
    
    if cuda:
        loss = loss.cuda()
        criterion = criterion.cuda()

    print('===> Setting Optimizer')
    optimizer = optim.Adagrad(model.parameters(), lr = lr, weight_decay = opt.weight_decay)

    if opt.resume:
        if os.path.exists(opt.resume):
            checkpoints = torch.load(opt.resume)
            model.load_state_dict(checkpoints['model'])
            optimizer.load_state_dict(checkpoints['optimizer'])
            opt.start_epoch = checkpoints['epoch'] + 1
            lr = checkpoints['lr']
            print("continue to train on {} Channel from Epoch {}".format(opt.channel, opt.start_epoch))
        else:
            raise Exception("The path of pre_trained is wrong")
    else:
        print("the first training on {} Channel".format(opt.channel))


    
    print('===> training model')
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        train(training_data_loader, model, optimizer, loss, epoch)
        val(val_data_loader, model, criterion)
        if epoch % 10 ==0:  
            checkpoint(model, optimizer, epoch, lr, opt.channel)

        data_frame = pd.DataFrame(
        data={ 
            'Avg. Loss': results['Avg. Loss'],
            'Avg. PSNR1': results['Avg. PSNR1'],
            'Avg. PSNR2': results['Avg. PSNR2'],
            'Avg. SSIM1': results['Avg. SSIM1'],
            'Avg. SSIM2': results['Avg. SSIM2'],
            },
        index=range(opt.start_epoch, epoch + 1))
        data_frame.to_csv('./result-{}-{}.csv'.format(opt.channel, opt.start_epoch), index_label='Epoch')


if __name__ == '__main__':
    main()
