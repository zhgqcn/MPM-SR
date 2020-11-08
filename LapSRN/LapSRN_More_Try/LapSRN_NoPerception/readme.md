## 基于我的模型的基础上，对Loss进行修改

- 仅仅把`Loss`中的`Perception Loss`删除，这样子修改的目的是：
  - `Perception Loss`是利用`Vgg16`网络进行特征提取后进行计算的，我想让loss更加直接以来于RGB计算结果与Target直接的差别
  
![Snipaste_2020-11-08_09-50-20](https://tva2.sinaimg.cn/large/005tpOh1ly1gkhiz67j4cj31hc0swn36.jpg)

  
  
# Loss的代码如下
```python
import torch
from torch import nn
from torchvision.models.vgg import vgg16

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Image Loss
        image_loss = CharbonnierLoss(out_images, target_images)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # return image_loss + 1e-8 * perception_loss + 2e-8 * tv_loss
        return image_loss + 1e-8 * tv_loss  # 修改了这一行代码


def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-6)) # epsilon=1e-3


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

```
