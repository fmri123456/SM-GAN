import os
import scipy.io as sio
import torch.nn as nn
import utils
import torch
from generator import generator
from discriminator import discriminator
from discriminator2 import discriminator2
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不用gpu，cuda有点问题

# 加载数据，包括AD、EMCI、LMCI
dataFile1 = 'SNP.mat'
dataFile2 = 'fMRI.mat'
data1 = sio.loadmat(dataFile1)
data2 = sio.loadmat(dataFile2)
SNP = data1['feat']  # 633*45*70
fMRI = data2['feat']  # 633*116*70
print("数据加载完成...")
# 数据切分
EMCI_gene = SNP[0:197]  # 197*45*70
EMCI_time = fMRI[0:197]  # 197*116*70
LMCI_gene = SNP[197:400]  # 203*45*70
LMCI_time = fMRI[197:400]  # 203*116*70
AD_gene = SNP[400:633]  # 233*45*70
AD_time = fMRI[400:633]  # 233*116*70

# dataFile = 'input.mat'
# data = sio.loadmat(dataFile)
# AD_gene = data['AD_gene']  # 233*45*70
# AD_time = data['AD_time']  # 233*116*70
# EMCI_gene = data['EMCI_gene']  # 197*48*70
# EMCI_time = data['EMCI_time']  # 197*116*70
# LMCI_gene = data['LMCI_gene']  # 203*45*70
# LMCI_time = data['LMCI_time']  # 203*116*70
print("数据加载完成...")
# 数据切分
EMCI_gene = EMCI_gene[:, :45, :]  # 197*45*70

# 划分训练集和测试集(按照8：2的比例划分)
hum1 = EMCI_gene.shape[0]  # 197
partition_threshold1 = int(hum1 * 0.8)  # 157
shuffle_idx1 = np.array(range(0, hum1))
np.random.shuffle(shuffle_idx1)
EMCI_gene = EMCI_gene[shuffle_idx1]
EMCI_time = EMCI_time[shuffle_idx1]
Label_EMCI = np.zeros(partition_threshold1)

hum2 = LMCI_gene.shape[0]  # 203
partition_threshold2 = int(hum2 * 0.8)  # 162
shuffle_idx2 = np.array(range(0, hum2))
np.random.shuffle(shuffle_idx2)
LMCI_gene = LMCI_gene[shuffle_idx2]
LMCI_time = LMCI_time[shuffle_idx2]
Label_LMCI = np.ones(partition_threshold2)

hum3 = AD_gene.shape[0]  # 233
partition_threshold3 = int(hum3 * 0.8)  # 186
shuffle_idx3 = np.array(range(0, hum3))
np.random.shuffle(shuffle_idx3)
AD_gene = AD_gene[shuffle_idx3]
AD_time = AD_time[shuffle_idx3]
Label_AD = 2 * np.ones(partition_threshold3)

Labels = np.concatenate((Label_EMCI, Label_LMCI, Label_AD))
Labels = torch.LongTensor(Labels)
EMCI_gene_train = EMCI_gene[:partition_threshold1]
EMCI_gene_test = EMCI_gene[partition_threshold1:]
EMCI_time_train = EMCI_time[:partition_threshold1]
EMCI_time_test = EMCI_time[partition_threshold1:]

LMCI_gene_train = LMCI_gene[:partition_threshold2]
LMCI_gene_test = LMCI_gene[partition_threshold2:]
LMCI_time_train = LMCI_time[:partition_threshold2]
LMCI_time_test = LMCI_time[partition_threshold2:]

AD_gene_train = AD_gene[:partition_threshold3]
AD_gene_test = AD_gene[partition_threshold3:]
AD_time_train = AD_time[:partition_threshold3]
AD_time_test = AD_time[partition_threshold3:]
print("训练集划分完成...")
# 数据合并
gene_train = np.concatenate((EMCI_gene_train, LMCI_gene_train, AD_gene_train), axis=0)  # 505*45*70
time_train = np.concatenate((EMCI_time_train, LMCI_time_train, AD_time_train), axis=0)  # 505*116*70
gene_test = np.concatenate((EMCI_gene_test, LMCI_gene_test, AD_gene_test), axis=0)  # 128*45*70
time_test = np.concatenate((EMCI_time_test, LMCI_time_test, AD_time_test), axis=0)  # 128*116*70
# 转为张量
gene_train = torch.FloatTensor(gene_train)
gene_test = torch.FloatTensor(gene_test)
time_train = torch.FloatTensor(time_train)
time_test = torch.FloatTensor(time_test)

# 计算权重矩阵
gene_train = utils.set_weight(gene_train)
gene_test = utils.set_weight(gene_test)
time_train = utils.set_weight(time_train)
time_test = utils.set_weight(time_test)
EMCI_gene_test = gene_test[:40, :, :]
LMCI_gene_test = gene_test[40:81, :, :]
AD_gene_test = gene_test[81:, :, :]
EMCI_time_test = time_test[:40, :, :]
LMCI_time_test = time_test[40:81, :, :]
AD_time_test = time_test[81:, :, :]

torch.save(gene_train, 'gene_train.pth')
torch.save(gene_test, 'gene_test.pth')
torch.save(time_train, 'time_train.pth')
torch.save(time_test, 'time_test.pth')
torch.save(EMCI_gene_test, 'EMCI_gene_test.pth')
torch.save(LMCI_gene_test, 'LMCI_gene_test.pth')
torch.save(AD_gene_test, 'AD_gene_test.pth')
torch.save(EMCI_time_test, 'EMCI_time_test.pth')
torch.save(LMCI_time_test, 'LMCI_time_test.pth')
torch.save(AD_time_test, 'AD_time_test.pth')

# gene_train = torch.load('gene_train.pth')
# gene_test = torch.load('gene_test.pth')
# time_train = torch.load('time_train.pth')
# time_test = torch.load('time_test.pth')
# EMCI_gene_test = torch.load('EMCI_gene_test.pth')
# LMCI_gene_test = torch.load('LMCI_gene_test.pth')
# AD_gene_test = torch.load('AD_gene_test.pth')
# EMCI_time_test = torch.load('EMCI_time_test.pth')
# LMCI_time_test = torch.load('LMCI_time_test.pth')
# AD_time_test = torch.load('AD_time_test.pth')
print("权重矩阵计算完成...")


# 其他参数定义
LR = 0.0001
EPOCH = 8
batch_size = 32

# 构建数据集
dataset = TensorDataset(gene_train, time_train, Labels)
# 加载数据集
dataload = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
print("数据导入成功！")

G = generator()
D = discriminator()
D2 = discriminator2()

# 损失函数：二进制交叉熵损失(既可用于二分类，又可用于多分类)
criterion = nn.CrossEntropyLoss()
# 生成器的优化器
g_optimizer = torch.optim.Adam(G.parameters(), lr=LR)
# 判定器的优化器
d_optimizer = torch.optim.Adam(D.parameters(), lr=LR)

# 开始训练
print("开始训练...")
max_acc = 0
D_loss = []
G_loss = []
acc_list = []
for epoch in range(EPOCH):
    for step, (gene_train, time_train, Labels) in enumerate(dataload):
        print('第{}次训练第{}批数据'.format(epoch + 1, step + 1))
        num_img = gene_train.size(0)
        G_time = G(gene_train)
        # 定义标签
        label = np.concatenate((np.zeros(num_img), np.ones(num_img)))
        label = utils.onehot_encode(label)
        label = torch.LongTensor(np.where(label)[1])
        fake_label = label[0:num_img]  # 定义假label为0
        real_label = label[num_img:2 * num_img]  # 定义真实label为1
        # 计算损失
        real_out = D(time_train)
        d_loss_real = criterion(real_out, real_label)
        fake_out = D(G_time)
        d_loss_fake = criterion(fake_out, fake_label)
        # 训练分类
        Time = torch.cat((time_train, G_time), axis=0)
        Label = torch.cat((Labels, Labels))
        out1 = D2(Time)
        d2_loss = criterion(out1, Label)

        # 反向传播和优化
        d_loss = d_loss_real + d_loss_fake + d2_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # 训练生成器
        gen = G(gene_train)
        fake_output = D(gen)
        out2 = D2(gen)
        g_loss1 = criterion(fake_output, real_label)
        g_loss2 = criterion(out2, Labels)
        g_loss = g_loss1 + g_loss2
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        # 保存每批数据的损失
        D_loss.append(d_loss)
        G_loss.append(g_loss)
        print('D_loss={:.4f}, G_loss={:.4f}'.format(sum(D_loss)/len(D_loss), sum(G_loss)/len(G_loss)))
    # 每10个为一个epoch
    if epoch % 10 == 0:
        num_test = gene_test.shape[0]
        G.eval()
        # 定义标签
        label = np.concatenate((np.zeros(num_test), np.ones(num_test)))
        label = utils.onehot_encode(label)
        label = torch.LongTensor(np.where(label)[1])
        fake_label = label[:num_test]  # 定义假label为0
        real_label = label[num_test:]  # 定义真实label为1
        gen_test = G(gene_test)
        output = D(gen_test)
        acc_val = utils.accuracy(output, fake_label)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Test set results:",
              "accuracy= {:.4f}".format(acc_val.item()))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        acc_list.append(float(acc_val.item()))

print("best accuracy={:.4f}".format(max_acc))
# 保存
torch.save(G.state_dict(), 'generator.pth')
torch.save(D.state_dict(), 'discriminator.pth')
torch.save(D2.state_dict(), 'discriminator2.pth')
# 载入
# G = generator()
# G.load_state_dict(torch.load('generator.pth'))
# D2 = discriminator2()
# D2.load_state_dict(torch.load('discriminator2.pth'))
# =====================================疾病状态识别
# 标准脑网络
std_EMCI_time = torch.mean(EMCI_time_test, dim=0)
std_LMCI_time = torch.mean(LMCI_time_test, dim=0)
std_AD_time = torch.mean(AD_time_test, dim=0)
# 生成脑网络与标准脑网络的相似性值C
generator_time = G(gene_test)
norm_C = D2(generator_time)
# 状态识别指标计算(ACC、Precision、Recall、F1_score)
EMCI_hum = EMCI_gene_test.size(0)
LMCI_hum = LMCI_time_test.size(0)
AD_hum = AD_time_test.size(0)
sum_hum = EMCI_hum + LMCI_hum + AD_hum
identify_labels = torch.cat((torch.zeros(EMCI_hum), torch.ones(LMCI_hum), 2 * torch.ones(AD_hum)))
identify_ACC, Precision, Recall = utils.identify_indicators(norm_C, identify_labels)
# 加权平均
identify_Precision = (EMCI_hum / sum_hum) * Precision[0] + (LMCI_hum / sum_hum) * Precision[1] + (AD_hum / sum_hum) * Precision[2]
identify_Recall = (EMCI_hum / sum_hum) * Recall[0] + (LMCI_hum / sum_hum) * Recall[1] + (AD_hum / sum_hum) * Recall[2]
# 计算F1-score
identify_F1_score = (2 * identify_Precision * identify_Recall) / (identify_Precision + identify_Recall)
print("identify_ACC:", identify_ACC)
print("identify_Precision:", identify_Precision)
print("identify_Recall:", identify_Recall)
print("identify_F1_score:", identify_F1_score)


# =====================================疾病风险预测
dataFile3 = 'EMCI_LMCI_SNP.mat'
dataFile4 = 'LMCI_AD_SNP.mat'
data3 = sio.loadmat(dataFile3)
data4 = sio.loadmat(dataFile4)
LMCI_true = data3['LMCI_true']  # 33*45*70
LMCI_fake = data3['LMCI_fake']  # 33*45*70
AD_true = data4['AD_true']  # 48*45*70
AD_fake = data4['AD_fake']  # 48*45*70
LMCI = np.concatenate((LMCI_true, LMCI_fake), axis=0)  # 66*45*70
AD = np.concatenate((AD_true, AD_fake), axis=0)  # 96*45*70
LMCI = torch.FloatTensor(LMCI)
AD = torch.FloatTensor(AD)
LMCI = utils.set_weight(LMCI)
AD = utils.set_weight(AD)
generator_EMCI_time = G(LMCI)
generator_LMCI_time = G(AD)
# 生成脑网络与下一疾病状态的标准脑网络相似度S
predict_LMCI = utils.predict(generator_EMCI_time, std_LMCI_time)
predict_AD = utils.predict(generator_LMCI_time, std_AD_time)
print("predict_LMCI:", predict_LMCI)
print("predict_AD:", predict_AD)
# 风险预测指标计算(ACC、SEN、SPE、BAC)
count1 = LMCI_true.shape[0]
count2 = AD_true.shape[0]
Based_EMCI_labels = torch.cat((torch.ones(count1), torch.zeros(count1)))
Based_LMCI_labels = torch.cat((torch.ones(count2), torch.zeros(count2)))
# 中位数
num1 = torch.median(predict_LMCI)
num2 = torch.median(predict_AD)
Based_EMCI_ACC, Based_EMCI_SEN, Based_EMCI_SPE, Based_EMCI_BAC = utils.predict_indicators(predict_LMCI, Based_EMCI_labels, num1)
Based_LMCI_ACC, Based_LMCI_SEN, Based_LMCI_SPE, Based_LMCI_BAC = utils.predict_indicators(predict_AD, Based_LMCI_labels, num2)
print("Based_EMCI_ACC:", Based_EMCI_ACC)
print("Based_EMCI_SEN:", Based_EMCI_SEN)
print("Based_EMCI_SPE:", Based_EMCI_SPE)
print("Based_EMCI_BAC:", Based_EMCI_BAC)
print("Based_LMCI_ACC:", Based_LMCI_ACC)
print("Based_LMCI_SEN:", Based_LMCI_SEN)
print("Based_LMCI_SPE:", Based_LMCI_SPE)
print("Based_LMCI_BAC:", Based_LMCI_BAC)

