import torch.nn as nn
import torch

class MemoryModule(nn.Module):
    def __init__(self, memory_size, memory_dim):
        super(MemoryModule, self).__init__()
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))  # 可学习的记忆矩阵

    def forward(self, query):
        # 简单的记忆读取：计算查询与记忆的相似度并返回加权记忆
        similarity = torch.matmul(query, self.memory.t())
        attention_weights = torch.softmax(similarity, dim=-1)
        memory_out = torch.matmul(attention_weights, self.memory)
        return memory_out

if __name__ == '__main__':
    model = MemoryModule(100,20)
    input = torch.randn(64,50,3)
    ouput = model(input)
    print(ouput.shape)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 假设有一个原始数据框
np.random.seed(40)  # 设置随机种子以确保可复现
df = pd.read_csv("../data/CVT/test.csv")

n = df.shape[0]

# 随机选择多个误差时间段
num_error_segments = np.random.randint(200, 210)  # 随机选择误差段数，200到250段

# 用于标记误差的列
df['label'] = 0

# 存储已使用的误差段区间
occupied_intervals = []

# 设置每列发生误差的概率
error_probabilities = {'a': 0.6, 'b': 0.5, 'c': 0.3}  # 'a' 列 80% 概率，'b' 列 50% 概率，'c' 列 30% 概率

# 生成误差段
error_segments = []
for _ in range(num_error_segments):
    # 随机选择误差段的开始位置和持续时间
    error_start = np.random.randint(0, n - 100)  # 错误开始位置
    error_length = np.random.randint(50, 60)  # 错误持续时间

    # 记录这个误差段
    occupied_intervals.append((error_start, error_length))

    # 根据设置的概率选择需要施加误差的列
    columns_to_error = []
    for col, prob in error_probabilities.items():
        if np.random.rand() < prob:  # 根据概率决定是否施加误差
            columns_to_error.append(col)

    # 随机选择误差幅度，在 0.2% 到 0.3% 之间浮动
    error_percentage = np.random.uniform(0.003, 0.005)

    # 记录误差段
    error_segments.append((error_start, error_length, columns_to_error, error_percentage))

# 对每个误差段施加误差
for error_start, error_length, columns_to_error, error_percentage in error_segments:
    for col in columns_to_error:
        df.loc[error_start:error_start + error_length, [col]] *= (1 + error_percentage)
    df.loc[error_start:error_start + error_length, 'label'] = 1