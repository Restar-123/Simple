import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryModule(nn.Module):
    def __init__(self, N, H):
        """
        记忆模块，包含 N 个记忆项，每个记忆项的维度为 H。
        参数:
            N (int): 记忆项的数量。
            H (int): 每个记忆项的维度。
        """
        super(MemoryModule, self).__init__()
        self.memory_items = nn.Parameter(torch.randn(N, H))  # 随机初始化 N 个记忆项
        self.N = N  # 记忆项数量
        self.H = H  # 每个记忆项的维度

    def forward(self, z):
        """
        前向传播，包括记忆检索和更新操作。
        参数:
            z (torch.Tensor): 输入的特征表示，形状为 (B, H)，
                              其中 B 是批量大小，H 是特征维度。
        返回:
            z_hat (torch.Tensor): 更新后的特征表示，形状为 (B, H)。
        """
        # 对输入 z 和记忆项进行归一化，计算余弦相似性
        z_norm = F.normalize(z, dim=1)  # 对 z 进行 L2 范数归一化，形状为 (B, H)
        memory_norm = F.normalize(self.memory_items, dim=1)  # 对记忆项归一化，形状为 (N, H)

        # 记忆检索
        similarity = torch.matmul(z_norm, memory_norm.T)  # 计算相似度矩阵，形状为 (B, N)
        weights = F.softmax(similarity, dim=1)  # 通过 softmax 转化为匹配分数，形状为 (B, N)
        z_hat = torch.matmul(weights, self.memory_items)  # 根据权重加权记忆项，得到更新后的特征表示 z_hat，形状为 (B, H)

        # 记忆更新
        with torch.no_grad():  # 更新操作不参与梯度计算
            for i in range(self.N):
                # 找到与记忆项 i 最接近的输入特征
                relevant_indices = weights[:, i] >= weights.max(dim=1).values  # 获取最近邻样本的布尔索引，形状为 (B,)
                relevant_z = z[relevant_indices]  # 筛选出的最近邻样本集合，形状为 (|Ui|, H)
                if relevant_z.size(0) > 0:  # 确保存在最近邻样本
                    relevant_similarity = similarity[relevant_indices, i]  # 获取对应相似度，形状为 (|Ui|,)
                    update_weights = F.softmax(relevant_similarity, dim=0)  # 计算更新权重，形状为 (|Ui|,)
                    update_term = torch.sum(update_weights.unsqueeze(1) * relevant_z, dim=0)  # 更新项，形状为 (H,)
                    self.memory_items[i] += update_term  # 更新记忆项
                    self.memory_items[i] = F.normalize(self.memory_items[i], dim=0)  # 再次归一化记忆项

        return z_hat  # 返回更新后的特征表示

# 示例使用
if __name__ == "__main__":
    batch_size = 64  # 输入特征的批量大小
    feature_dim = 3  # 输入特征的维度
    memory_size = 100  # 记忆模块中记忆项的数量

    memory_module = MemoryModule(N=memory_size, H=feature_dim)  # 实例化记忆模块
    input_features = torch.randn(batch_size, feature_dim)  # 随机生成输入特征

    output_features = memory_module(input_features)  # 前向传播，输出更新后的特征
    print("Output features:", output_features.shape)  # 打印输出特征的形状
