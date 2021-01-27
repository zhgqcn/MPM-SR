# 这个仓库用于记录学长的原始的网络模型,即LapSRN_v1.0

## 该模型要点：
- 使用单通道训练，对于不同的通道训练不同的模型，预处理和后处理操作比较麻烦

- 网络结构
![Snipaste_2020-11-05_09-03-50](https://tva1.sinaimg.cn/large/005tpOh1ly1gke0t15jxfj30l90fu0v3.jpg)

- 训练结果
![Snipaste_2020-11-05_09-05-43](https://tva1.sinaimg.cn/large/005tpOh1ly1gke0tsdojej30za0ngwgk.jpg)

- 测试效果 
![-11-05_09-06-26](https://tva2.sinaimg.cn/large/005tpOh1ly1gke0uht635j30uj0j3grx.jpg)
https://tva2.sinaimg.cn/large/005tpOh1ly1gke0uht635j30uj0j3grx.jpg
- X4结果图
下载 HRW.tif


## 训练须知
1. 不同的通道利用不同的训练方式可获得更好的效果  

R通道	400	0.0455	28.06	27.29   利用DataSet_less迭代训练  

G通道	220	0.0335	32.38	31.72   直接通过原始图像，利用centercrop 64 128 256  
