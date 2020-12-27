
# LapSRN_v3
## Main Idea
> Loss分为两步优化
```python
        loss_g_X2 = Loss_g(HR_2_g, HR_2_g_Target)
        loss_g_X4 = Loss_g(HR_4_g, HR_4_g_Target)
        loss_g = loss_g_X2 + loss_g_X4
        epoch_loss_g += loss_g.item()
        loss_g_X2.backward(retain_graph = True)
        loss_g.backward()
```
![Snipaste_2020-12-27_10-42-43](https://tvax1.sinaimg.cn/large/005tpOh1ly1gm27v61foqj31h70mjgpe.jpg)

## LapSRN_v3.0
1. 把R通道和G通道**集成**在同一份代码里，原来的是想这样子可以使用一台服务器进行训练，然后其他的可以训练别的模型。
但是集成后训练**非常慢**，而且手里也有空闲的服务器，还不如分开进行训练。Emmmm
2. ***test_util把测试使用的代码弄成了函数，可供后面测试的方便调用***


## LapSRN_v3.1
1. 因为v3.0版本的原因，进行了拆分训练
