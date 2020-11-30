# 建立仓库版本过程说明
[Experiment1.0（即LapSRN仓库）](https://github.com/zhgqcn/GraduationProject/tree/master/LapSRN)  ➡  [Experiment2.0](https://github.com/zhgqcn/GraduationProject/tree/master/Experiment2.0)
# 其他仓库
- [ResultCompare](https://github.com/zhgqcn/GraduationProject/tree/master/ResultCompare) 用于存放超分结果
- [notes](https://github.com/zhgqcn/GraduationProject/tree/master/notes)  用于存放笔记
- [Pretreatment](https://github.com/zhgqcn/GraduationProject/tree/master/Pretreatment) 用于存放预处理和后处理小程序
# 模型更替过程
## LapSRN版本更替
[LapSRN_v1.0](https://github.com/zhgqcn/GraduationProject/tree/master/LapSRN/LapSRN_HRW) **→**  [LapSRN_v1.1](https://github.com/zhgqcn/GraduationProject/tree/master/LapSRN/LapSRN_HRW_Adagrad)

## SiameseLapSRN版本更替


# Task 
1. 控制单一变量1118
2. 只用其中一部分模型


## 论文
- ESPCN，FSRCNN，VDSR，SRResNet，这些文章都有可取之处，在刷榜的同时带给了领域新的认识。比如说 ESPCN 之后大家都开始用 `Pixel Shuffle 上采样`，FSRCNN 之后大家都开始在`最后再做上采样`，VDSR 开始`残差`被引入，到 SRResNet 引入`残差 Block`。这些工作虽然是在刷榜，但是我并不认为他们是在灌水，只是当时的领域的认知停留在刷 PSNR 而已。然后 `SRGAN` 提出，学术界尤其是工业界开始发现刷 PSNR 其实没什么用，因为 PSNR 高的超分图像看起来非常的不真实，根本没法用在产品上，有一部分人开始宣传做基于`感知`的超分辨，到去年 ECCV PIRM SR 比赛，一批做感知的论文出来，继续在这个方向持续的做努力。这其中的一些是在推进整个领域的发展的，我同样也不认为他们是在灌水。而与此同时，还在做网络结构的论文，灌水的比例就提高了，乃至于到现在审稿的时候看到做网络结构的，基本都水的不行。这也比较好理解，真正做研究的都转移战场了，**这个时候入场的大多都是新人了**。[引文出处](https://www.zhihu.com/question/324809101/answer/705103991)

- 如何收集, 或创造更接近于真实场景的训练数据[click Here](https://www.zhihu.com/question/293828312/answer/788366046)

- 真实世界图片超分辨率[click here](https://zhuanlan.zhihu.com/p/281201244)

- 超分辨技术简介[click here](https://zhuanlan.zhihu.com/p/263008440)

- 一篇图像超分辨率综述的阅读笔记[click here](https://zhuanlan.zhihu.com/p/276027388）

- 
