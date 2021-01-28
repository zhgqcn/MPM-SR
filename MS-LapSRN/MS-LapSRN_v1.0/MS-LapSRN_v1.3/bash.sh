
# r channel
## first train
python 'train.py' --channel 'r' --dataset '../DataSet_test/' --nEpochs 100 --cuda
## continue train
python "train.py" --channel 'r' --dataset "../DataSet_test/" --nEpochs 200 --cuda \
--resume './LapSRN_model_epoch_r_60.pth'


# g channel
## first train
python 'train.py' --channel 'g' --dataset '../DataSet_test/' --nEpochs 100 --cuda
## continue train
python "train.py" --channel 'g' --dataset "../DataSet_test/" --nEpochs 200 --cuda \
--resume './LapSRN_model_epoch_g_60.pth'