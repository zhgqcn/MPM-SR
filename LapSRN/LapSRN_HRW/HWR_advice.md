1. `1-LapSRN`文件夹针对R、G、B通道分离，对应`LapSRN-r`、`LapSRN-g`、`LapSRN-b`子文件夹，是可以对`Charbonnier loss`、`perceptual loss`、`TV loss`
赋予不同的权重(loss.py文件第23行)，指导网络训练。由于引入`VGG-16`预训练网络，需要三通道图片，可以使用`torchvision.transforms`中的`Lambda`，通过
`Lambda(lambda x: x.repeat(3,1,1))`，单通道图片复制使其达到3通道(data.py文件第11、17、23行和test.py第29行)，同时对网络输入和输出的通道数把1改成3(model.py文件第48、75、77行)。  

2. `2-LapSRN`文件夹针对R、G、B通道分离，对应`LapSRN-r`、`LapSRN-g`、`LapSRN-b`子文件夹，是`Charbonnier loss`指导网络训练。  

3. 在读取数据集部分，一定要注意低分辨率图像和参考的高分辨率图像是否对应上，如果没对应，可以将各自读取到列表进行有序化，即顺序读取文件夹下名称有序
的文件(“1-LapSRN”中dataset.py文件第20、23、26行或“2-LapSRN”中dataset.py文件第20、23、26行)。  

4. 训练的时候，损失值和PSNR值可以用字典和pandas进行记录和读入表格，最后再利用表格里面的绘制损失或PSNR曲线(`1-LapSRN`或`2-LapSRN`的plot.py文件)。  

5. 数据集的图片，已经按代码要求的那样放入`1-LapSRN`文件夹中的`dataset`子文件夹下相应的子文件夹。`2-LapSRN`文件夹中的`dataset`子文件夹下相应的
子文件夹已经按代码要求创建好，但未放入图片，可以直接从“1-LapSRN”文件夹中的“dataset”子文件夹下相应的子文件夹里的图片复制过来。
