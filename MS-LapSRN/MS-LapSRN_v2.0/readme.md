# 图像预处理

## 基于OpenCV的图像预处理

弄了很久，结果发现代码使用的是`PIL`加载并处理图像的，而OpenCV的数据格式和PIL的不同。所以当作学习了，这部分的学习比较直接同步到博客了: [OpenCV：图像增亮与直方图均衡](https://www.cnblogs.com/zgqcn/p/14287235.html)

## 基于PIL的图像预处理

### 图像增亮

```python
enh_bri = ImageEnhance.Brightness(image)
brightness = 2.5
image_brightened = enh_bri.enhance(brightness)
image_brightened.save('2.5-lightup' + '.tif')
```

![Snipaste_2021-01-16_21-01-21](https://tva4.sinaimg.cn/large/005tpOh1ly1gmpue96dnpj31ha0r7qnc.jpg)

通过多次尝试，感觉将参数设置为2.0比较合适，既不会太过亮，又相比于原图有更多的细节

### 对比度增强

```
enh_con = ImageEnhance.Contrast(image_brightened)
contrast = 2.0
image_contrasted = enh_con.enhance(contrast)
image_contrasted.save('brightness_' + str(brightness) + '_contrastUp_' + str(contrast) + '.tif')
```

![Snipaste_2021-01-16_22-30-18](https://tva4.sinaimg.cn/large/005tpOh1ly1gmpwp7swqbj31ha0r6x3i.jpg)

选择2.0的对比度比较适中

### 锐化增强

```
enh_sha = ImageEnhance.Sharpness(image_contrasted)
sharpness = 2.0
image_sharped = enh_sha.enhance(sharpness)
image_sharped.save('brightness_' + str(brightness) + '_contrastUp_' + str(contrast) + '-moreSharp2.0' + '.tif')
```

![Snipaste_2021-01-16_22-34-32](https://tvax4.sinaimg.cn/large/005tpOh1ly1gmpwtnl7gwj31ha0s4kfu.jpg)

锐化会使得噪声更加明显，所以就先不使用

### 预处理后结果

![Snipaste_2021-01-16_22-36-46](https://tva3.sinaimg.cn/large/005tpOh1ly1gmpwvztn6vj31ha0r7tuf.jpg)

<center><font color='green'>弹性纤维预处理后-前对比</font></center>

![Snipaste_2021-01-16_22-39-19](https://tva4.sinaimg.cn/large/005tpOh1ly1gmpwylg4l0j31ha0r8nb6.jpg)

<center><font color='green'>胶原纤维预处理后-前对比</font></center>

## 存在的问题

```python
def image_augument(image, brightness=2.0, contrast=2.0):  
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened = enh_bri.enhance(brightness)
    enh_con = ImageEnhance.Contrast(image_brightened)
    image_contrasted = enh_con.enhance(contrast)
    return  image_contrasted
```

由于图像增亮和对比度拉伸都用相同的参数进行调节，所以对于所有的数据可能无法每一张都除了的恰到好处，可能会使得有些亮度及对比度过于明显

![Snipaste_2021-01-17_23-10-24](https://tvax1.sinaimg.cn/large/005tpOh1ly1gmr3h6xvltj30o8076t9v.jpg)

<center><font color='green'>亮度过大，本来应该内圈亮一点</font></center>

# 网络训练

![Snipaste_2021-01-17_23-19-48](https://tva4.sinaimg.cn/large/005tpOh1ly1gmr3r0xcp6j31490g9jt2.jpg)

<center><font color='green'>基于数据增强设计两组实验</font></center>

## 视觉对比

![Snipaste_2021-01-17_23-29-19](https://tva2.sinaimg.cn/large/005tpOh1ly1gmr40vbwbxj30qk0jntcz.jpg)

<center><font color='green'>实验结果对比</font></center>

## 量化对比

### PSNR和SSIM

|       版本       | PSNR_X4 | SSIM_X4 |
| :--------------: | :-----: | :-----: |
| `MS-LapSRN_v1.2` |  27.4   |  0.67   |
| `MS-LapSRN_v2.0` |  17.29  |  0.44   |
| `MS-LapSRN_v2.1` |  17.02  |  0.43   |

图像增强处理可能导致数据的分布出现偏差，导致量化计算偏低

### 直方图

![Snipaste_2021-01-17_23-38-54](https://tva4.sinaimg.cn/large/005tpOh1ly1gmr4avs1lyj30qq05st93.jpg)

`MS-LapSRN_v2.0`的直方图与`v1.2`的比较接近，而`v2.1`的直方图出现了较大的偏差（**这个是不是数据量变少了？**）
