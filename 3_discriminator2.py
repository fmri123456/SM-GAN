import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class discriminator2(nn.Module):
    def __init__(self):
        super(discriminator2, self).__init__()  # 初始化
        self.linear = nn.Sequential(nn.Linear(116 * 116, 116),
                                    nn.ReLU(),
                                    nn.Linear(116, 3))

    # 这里的x表示真实/生成脑网络
    def forward(self, x):
        # 平铺
        x = torch.flatten(x, 1, -1)
        # 全连接层
        x = self.linear(x)
        x = F.softmax(x)
        return x