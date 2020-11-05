# 双通道`LapSRN`网络

![Snipaste_2020-11-02_00-03-15](https://tvax2.sinaimg.cn/large/005tpOh1ly1gka4afillrj30y60ks0v5.jpg)

1. 重构`LapSRN`网络，可以实现端到端训练，省去了预处理和后处理的麻烦
2. `Loss`的计算由单通道的计算转换为`RGB`的计算

3. 损失函数修改了`SGD`为`Adagrad`
4. 网络收敛很快，训练`20Epochs`都能有不错的效果

5. 训练指标

| Epochs |  Loss  | PSNR_X2 | PSNR_X4 |
| :----: | :----: | :-----: | :-----: |
|  210   | 0.0238 |  30.67  |  29.51  |

6. 待完善：

- 封装
- 目前只支持`batchSize = 1`，可以通过`transpose`修改

7. 我找了学长拿论文里的图作为一个比较：

![Snipaste_2020-11-02_00-29-21](https://tvax1.sinaimg.cn/large/005tpOh1ly1gka51o449vj30zy0iodmf.jpg)

原图放在压缩包，后缀说明：

- ALL：基于2112张训练集，迭代次数较少
- Less：基于166张训练集，可进行较多的迭代
- 默认的是基于Less训练的
