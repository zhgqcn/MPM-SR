# LapSRN_v4
## Main Idea
1. 把**代码模块化**（写成一个个函数，方便对代码进行修改）
2. 加入`SSIM LOSS`, **Idea From [LPNet](https://www.cnblogs.com/zgqcn/p/14048801.html)**

## LapSRN_v4.0
```python
  return image_loss + 1e-8 * perception_loss + 2e-8 * tv_loss + 0.5*ssim_loss
```

## LapSRN_v4.1
```python
  return image_loss +  0.5 * ssim_loss
```

## 模块后的训练于继续训练
1. 选择训练通道
2. 加入预训练模型`--resume`就可以继续训练
```
# g channel
## first train
python 'train.py' --channel 'g' --dataset '../DataSet_test/' --nEpochs 100 --cuda
## continue train
python "train.py" --channel 'g' --dataset "../DataSet_test/" --nEpochs 200 --cuda \
--resume './LapSRN_model_epoch_g_60.pth'
```
