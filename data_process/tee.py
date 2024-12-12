import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 假设有一个原始数据框
np.random.seed(40)  # 设置随机种子以确保可复现
n = 70000  # 数据有7万条
df = pd.DataFrame({
    'a': np.linspace(1, 100, n),
    'b': np.linspace(1, 100, n),
    'c': np.linspace(1, 100, n)
})



# 随机选择多个误差时间段
num_error_segments = np.random.randint(200, 250)  # 随机选择误差段数，200到250段
error_percentage = 0.002  # 0.2%

# 用于标记误差的列
df['label'] = 0

# 存储已使用的误差段区间
occupied_intervals = []

# 生成误差段
error_segments = []
for _ in range(num_error_segments):
    # 随机选择误差段的开始位置和持续时间
    error_start = np.random.randint(0, n - 100)  # 错误开始位置
    error_length = np.random.randint(50, 80)  # 错误持续时间

    # 确保误差段不会与已占用区间重叠
    while any(start <= error_start + error_length - 1 and start + length >= error_start for start, length in occupied_intervals):
        error_start = np.random.randint(0, n - 100)  # 重新选择开始位置

    # 记录这个误差段
    occupied_intervals.append((error_start, error_length))

    # 随机选择需要施加误差的列
    columns_to_error = np.random.choice(['a', 'b', 'c'], size=np.random.randint(1, 4), replace=False)
    error_segments.append((error_start, error_length, columns_to_error))

# 对每个误差段施加误差
for error_start, error_length, columns_to_error in error_segments:
    for col in columns_to_error:
        df.loc[error_start:error_start + error_length, [col]] *= (1 + error_percentage * np.random.uniform(-1, 1))
    df.loc[error_start:error_start + error_length, 'label'] = 1

# 绘制label列的折线图
plt.figure(figsize=(12, 6))
plt.plot(df['label'], color='orange', label='Label (Error Occurred)', linewidth=0.8)
plt.title('Label Column - Error Occurrence Over Time')
plt.xlabel('Index')
plt.ylabel('Label Value (0 = No Error, 1 = Error Occurred)')
plt.legend()
plt.grid(True)
plt.show()