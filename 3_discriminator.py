import torch
import torch.nn as nn
import torch.nn.functional as F
import generator_conv
from torch.nn.parameter import Parameter
import math
# 判定器模型设计
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()  # 初始化
        # 聚合系数矩阵θ
        self.theta = Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        # 比例系数α
        self.alpha = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.reset_parameters()
        self.linear = nn.Sequential(nn.Linear(116 * 116, 116),
                                    nn.ReLU(),
                                    nn.Linear(116, 2))
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)

    # 这里的x表示真实/生成脑网络
    def forward(self, x):
        A = generator_conv.set_A(x)
        P = generator_conv.set_P(A)
        E = torch.bmm(P, x) * self.theta  # 哈达玛积
        delta_E = generator_conv.set_delta_E(E)
        delta_W = generator_conv.set_delta_W(delta_E)
        # 更新权重矩阵new_W
        new_W = self.alpha * delta_W + x
        # 平铺
        x = torch.flatten(new_W, 1, -1)
        # 全连接层
        x = self.linear(x)
        x = F.softmax(x)
        return x
