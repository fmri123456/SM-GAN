import torch
import torch.nn as nn
import torch.nn.functional as F
import generator_conv
from torch.nn.parameter import Parameter
import math
# 生成器模型设计
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()  # 初始化
        # 可学习参数θ
        self.theta = Parameter(torch.FloatTensor(45, 45), requires_grad=True)
        # 比例系数α
        self.alpha = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.25)
        # 点映射矩阵Mv
        self.Mv = Parameter(torch.FloatTensor(116, 45), requires_grad=True)
        self.Mv.data.normal_(0, 1)
        # 点信息映射矩阵Ms
        self.Ms = Parameter(torch.FloatTensor(45, 116), requires_grad=True)
        self.Ms.data.normal_(0, 1)
        # 可学习参数λ
        self.Lambda = Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        # 比例系数β
        self.beta = Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.25)
        self.reset_parameters1()
        self.reset_parameters2()
        self.linear = nn.Sequential(nn.Linear(45*45, 116*45),
                                    nn.Linear(116*45, 116*116))

    def reset_parameters1(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)

    def reset_parameters2(self):
        stdv = 1. / math.sqrt(self.Lambda.size(1))
        self.Lambda.data.uniform_(-stdv, stdv)

    # 这里的x表示的是权重矩阵
    def forward(self, x):
        # ----------------------聚合卷积
        A = generator_conv.set_A(x)
        P = generator_conv.set_P(A)
        # 计算节点聚集的结构信息量矩阵E
        E = torch.bmm(P, x) * self.theta  # 哈达玛积
        delta_E = generator_conv.set_delta_E(E)
        delta_W = generator_conv.set_delta_W(delta_E)
        # 更新权重矩阵new_W
        new_W = self.alpha * delta_W + x
        # ----------------------映射结构
        # 节点映射矩阵
        Wv = torch.matmul(self.Mv, new_W)
        # 节点结构映射矩阵
        W_ = torch.matmul(Wv, self.Ms)
        # 对称化结构信息映射矩阵
        W = generator_conv.set_normal_W(W_)
        # ----------------------扩散卷积
        A_ = generator_conv.set_A(W)
        P_ = generator_conv.set_P(A_)
        # 计算节点聚集相邻节点扩散的结构信息量矩阵E_
        E_ = torch.bmm(W, P_) * self.Lambda
        delta_E_ = generator_conv.set_delta_E(E_)
        delta_W_ = generator_conv.set_delta_W(delta_E_)  # hum * 116 * 116
        new_W_ = self.beta * delta_W_ + W
        new_W_ = F.leaky_relu(new_W_)
        return new_W_
