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


