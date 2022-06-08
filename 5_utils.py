import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import scipy.stats as stats


def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get, labes)), dtype=np.int32)
    #print(labes_onehot.shape)
    return labes_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# 权重矩阵(皮尔逊相关系数)
def set_weight(X):
    X = X.detach().numpy()
    # 节点个数
    row = X.shape[0]
    # 特征个数
    col = X.shape[1]
    # 初始化权重矩阵
    W = np.zeros((row, col, col))
    for k in range(row):
        for i in range(col):
            for j in range(col):
                # 主对角线元素为0
                if j != i:
                    # Pearson
                    W[k][i][j] = np.min(np.corrcoef(X[k][i], X[k][j]))
                    # Spearman
                    # W[k][i][j] = stats.spearmanr(X[k][i], X[k][j])[0]  # 返回两个值：correlation,pvalue。
                    # Kendall
                    # W[k][i][j] = stats.kendalltau(X[k][i], X[k][j])[0]  # 返回两个值：correlation,pvalue。
    W = weight_threshold(W)
    W = torch.FloatTensor(W)
    return W


# 权重矩阵阈值化(关联系数最小的20%元素置0)
def weight_threshold(W):
    row = W.shape[0]
    col = W.shape[1]
    result = np.zeros((row, col, col))
    for i in range(row):
        threshold = np.sort(np.abs(W[i].flatten()))[int(col * col * 0.2)]  # 阈值
        result[i] = W[i] * (np.abs(W[i]) >= threshold)
    return result

# 相似性计算(生成脑网络gen与标准脑网络std)
def similar(gen, std):
    gen = gen.detach().numpy()
    std = std.detach().numpy()
    row = gen.shape[0]
    col = gen.shape[1]
    C = np.zeros((row, 1))
    for i in range(row):
        G = gen[i].reshape(col, col)
        tmp = np.abs(G - std)
        temp = tmp ** 2
        C[i] = 1 / np.sum(temp)
    C = torch.FloatTensor(C)
    return C

# 风险预测P
def predict(gen, std):
    gen = gen.detach().numpy()
    std = std.detach().numpy()
    row = gen.shape[0]
    col = gen.shape[1]
    S = np.zeros((row, col, col))
    P = np.zeros(row)
    for k in range(row):
        temp = gen[k].reshape(col, col)
        for i in range(col):
            for j in range(col):
                if np.abs(temp[i, j] - std[i, j]) <= 0.1:
                    S[k, i, j] = 1
    for m in range(row):
        P[m] = np.sum(S[m]) / (col * (col - 1))
    P = torch.FloatTensor(P)
    return P


# 状态识别指标计算
def identify_indicators(output, labels):
    print(output.shape)
    row = output.size(1)
    matrix_array = torch.zeros(row, row)  # 创建矩阵，行代表真实标签，列代表预测标签
    indicator_array = torch.zeros(row, row)  # 创建矩阵，行代表类别，列代表指标(TP、TN、FP)
    Precision = torch.zeros(row)  # 保存每一类的Precision值
    Recall = torch.zeros(row)  # 保存每一类的Recall值
    for i in range(row):
        for j in range(row):
            # output.max(-1):返回每行最大的值和下标,[0]代表值，[1]代表下标
            matrix_array[i, j] = ((output.max(-1)[1] == j) & (labels == i)).sum()
    print("matrix_array:", matrix_array)
    for m in range(row):
        indicator_array[m, 0] = matrix_array[m, m]  # TP
        indicator_array[m, 1] = torch.sum(matrix_array[m]) - matrix_array[m, m]  # FN
        indicator_array[m, 2] = torch.sum(matrix_array[:, m]) - matrix_array[m, m]  # FP
    print("indicator_array:", indicator_array)
    ACC = torch.sum(torch.diagonal(matrix_array)) / torch.sum(matrix_array)
    for n in range(row):
        Precision[n] = indicator_array[n, 0] / (indicator_array[n, 0] + indicator_array[n, 2])
        Recall[n] = indicator_array[n, 0] / (indicator_array[n, 0] + indicator_array[n, 1])
    print("Precision:", Precision)
    print("Recall:", Recall)
    return ACC, Precision, Recall


# 风险预测指标计算
def predict_indicators(output, labels, num):
    output[output > num] = 1
    output[output <= num] = 0
    TP = ((output == 1) & (labels == 1)).sum()
    TN = ((output == 0) & (labels == 0)).sum()
    FP = ((output == 0) & (labels == 1)).sum()
    FN = ((output == 1) & (labels == 0)).sum()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    SEN = TP / (TP + FN)
    SPE = TN / (FP + TN)
    BAC = (SEN + SPE) / 2
    return ACC, SEN, SPE, BAC


