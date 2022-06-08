import numpy as np
import scipy.io as sio
import torch

dataFile = 'F://胡溪实验/LS_Run/0_输入数据.mat'  # 加载文件
dataFile1 = 'F://胡溪实验/0_输入数据.mat'
data = sio.loadmat(dataFile)
# 获取脑区基因网络的矩阵
Wei = data['w']
# 定义真实样本是否为患者的标签
label1 = np.concatenate((np.ones(233), np.zeros(237)))
# 定义生成样本是否为患者的标签
label2 = np.concatenate((np.ones(233), np.zeros(237)))
shuffle_idx = np.array(range(0, 470))
# 打乱标签
np.random.shuffle(shuffle_idx)
labels1 = label1[shuffle_idx]
labels2 = label2[shuffle_idx]
wei = Wei[shuffle_idx]
train_id = range(0, 420)
test_id = range(420, 470)


def load():
    return Wei


def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get, labes)), dtype=np.int32)
    return labes_onehot
