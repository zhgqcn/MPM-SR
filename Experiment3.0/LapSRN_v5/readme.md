# LapSRN_v5

## Main Idea
> 加入`DnCNN`去噪网络对`HR`进行去噪处理

## LapSRN_v5.0
```python
  return image_loss + 1e-8 * perception_loss + 2e-8 * tv_loss
```

## LapSRN_v5.1
```python
  return image_loss +  0.5 * ssim_loss
 ```
 
 ## 结果
 PSNR与SSIM都比之前高了，但是，视觉效果却没有提升　　
 
 |Epoch|Loss|PSNR2|PSNR4|SIM2|SIM4|
 |--|--|--|--|--|--|
 |280|0.1789|28.93|	28.85|	0.85|	0.87|
