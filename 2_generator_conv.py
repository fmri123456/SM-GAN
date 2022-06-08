import numpy as np
import torch

# 邻接矩阵A
def set_A(W):
    W = W.detach().numpy()
    row = W.shape[0]
    col = W.shape[1]
    A = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                if W[k, i, j] != 0:
                    A[k, i, j] = 1
    A = torch.FloatTensor(A)
    return A

# 聚合矩阵P
def set_P(A):
    A = A.detach().numpy()
    row = A.shape[0]
    col = A.shape[1]
    P = np.zeros((row, col, col))
    I = np.eye(col)
    for k in range(row):
        P[k] = A[k] + I
    P = torch.FloatTensor(P)
    return P

# 节点结构信息变化量矩阵delta_E
def set_delta_E(E):
    E = E.detach().numpy()
    row = E.shape[0]
    col = E.shape[1]
    delta_E = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            delta_E[k, i, i] = np.sum(E[k, i])
    delta_E = torch.FloatTensor(delta_E)
    return delta_E

# 边权重变化量矩阵delta_W
def set_delta_W(delta_E):
    delta_E = delta_E.detach().numpy()
    row = delta_E.shape[0]
    col = delta_E.shape[1]
    # 边结构传递矩阵F
    F = np.ones((col, col))
    # 边权重变化量矩阵
    delta_W = np.zeros((row, col, col))
    for k in range(col):
        F[k, k] = 0
    # 单个节点对边权重的扩散量delta_W_c
    delta_W_c = np.matmul(F, delta_E) / (col - 1)
    for i in range(row):
        delta_W[i] = delta_W_c[i].reshape(col, col) + delta_W_c[i].reshape(col, col).T
    delta_W = torch.FloatTensor(delta_W)
    return delta_W

# 对称化结构信息映射矩阵W
def set_normal_W(W_):
    W_ = W_.detach().numpy()
    row = W_.shape[0]
    col = W_.shape[1]  # 116
    normal_W = np.zeros((row, col, col))
    for k in range(row):
        A = W_[k].reshape(col, col)
        B = A.T
        normal_W[k] = (A + B) / 2
    normal_W = torch.FloatTensor(normal_W)
    return normal_W

