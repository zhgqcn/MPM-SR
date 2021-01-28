# MS-LapSRN

## 网络创新

该模型是**LapSRN**官方作者提出的新模型，在原来的基础上加入了<font color='red'>**局部残差ResNet**和**递归网络RecursiveNet**</font>

![](https://www.pianshen.com/images/343/90f13552ec80d045a918b1833ac80fdf.png)

<center><font color='green'>图1：原来的LapSRN</font></center>

![](https://www.pianshen.com/images/589/f0df4c665df85e60bc6918eba38b0125.png)

<center><font color='green'>图2：MS-LapSRN</font></center>



![](https://www.pianshen.com/images/572/d3d418a89be54dcf5b1425cc13688ac4.png)

<center><font color='green'>图3：MS-LapSRN中加入了递归模块和局部残差模块</font></center>

**由于加入了递归模块，整体参数量降低了四分之一, 现在利用512*512的LR进行CPU超分辨都可以直接部署在自己电脑上**

![Snipaste_2021-01-03_23-52-27](https://tvax2.sinaimg.cn/large/005tpOh1ly1gmay2hbaq7j30v708atad.jpg)

<center><font color='green'>图4：MS-LapSRN中参数量仅为222k</font></center>

## 实验设计

> 因为官方源码是matlab写的，网上pytorch的参考资料不多，所以找了唯一能用的一份进行了修改，使得与之前的网络模型的超参数和训练的通道结构相同

**训练了三个实验，都是基于Loss的修改**

### 实验一：Loss和最初的Loss相同

```python
 return image_loss + 1e-8 * perception_loss + 2e-8 * tv_loss  
```

### 实验二：在之前的基础上去除了  tv_loss

猜想：以往的训练都有些平滑，可能是tv_loss造成的

```python
 return image_loss + 1e-8 * perception_loss
```

### 实验三：在实验二的基础上加上 ssim_loss

```python
return image_loss + 1e-8 * perception_loss +0.5 * ssim_loss  1.2
```

## 结果对比

注意：绿色通道都是相同的，都是基于实验三训练的结果，主要比较红色

![Snipaste_2021-01-04_00-23-38](https://tvax4.sinaimg.cn/large/005tpOh1ly1gmayx4bqfrj31h80s443x.jpg)

<center><font color='green'>图5：实验一和实验二比较</font></center>

1. 实验一和实验二比较几乎一模一样，说明在此网络上的作用不大的可能，也可能是 **2e-8 * tv_loss  **权值太小所以影响不大

![Snipaste_2021-01-04_00-26-13](https://tvax2.sinaimg.cn/large/005tpOh1ly1gmayzso0kzj31ha0s4jwl.jpg)

<center><font color='green'>图6：实验二和实验三比较</font></center>

2. 实验三比实验二更好：亮度更亮，细节更好

![Snipaste_2021-01-04_00-31-01](https://tvax2.sinaimg.cn/large/005tpOh1ly1gmaz52wcgjj31ha0s4te6.jpg)

<center><font color='green'>图7：LapSRN_v2.0和实验三比较</font></center>

3. LapSRN_v2.0即：基于黄仁伟代码修改了优化器为Adagrad，数据集为全部的数据集

4. 实验三的绿色部分更好，不会太过平滑，且颜色更亮。实验三的红色边缘更加细致。



