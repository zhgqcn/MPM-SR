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
