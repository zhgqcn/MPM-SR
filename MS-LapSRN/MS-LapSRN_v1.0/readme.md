# MS-LapSRN_v1.0

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

### 实验四：两部优化Loss
```python
SR_2, SR_4 = model(LR)
loss1 = Loss(SR_2, HR_2_target)
loss2 = Loss(SR_4, HR_4_target)
loss = loss1 + loss2
epoch_loss += loss.item()
# backward by two steps
loss1.backward(retain_graph = True)
loss.backward()
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

5. 对于实验四中的两步走  

  - 在测试集上进行计算

  |                           | PSNR_X4 | SSIM_X4 |
  | :-----------------------: | :-----: | :-----: |
  |   **`MS-LapSRN_v1.2`**    | 29.4800 | 0.8304  |
  | **`MS-LapSRN_v1.3(now)`** | 29.4640 | 0.8298  |

  - 视觉效果

  ![Snipaste_2021-01-10_22-36-58](https://tvax1.sinaimg.cn/large/005tpOh1ly1gmiz704b4bj31hc0s4n2e.jpg)

  <center><font color='green'><b>图1 左：MS-LapSRN_v1.2 右：MS-LapSRN_v1.3</b></font></center>

  - 结论

    - 没有之前的`MS-LapSRN_v1.2`一部计算的效果好：
     - 目前这一版本会导致过于平滑
     - 在量化效果上，也不如之前

  ![Snipaste_2021-01-18_08-52-02](https://tva3.sinaimg.cn/large/005tpOh1ly1gmrkanxxc7j30bq08owf5.jpg)

