import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    一个 TCN Block，包括膨胀卷积、非线性激活和残差连接。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding=(kernel_size - 1) * dilation // 2  # 因果卷积等效的 padding
        )
        self.layer_norm = nn.BatchNorm1d(out_channels)  # 使用批归一化提高训练稳定性
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.layer_norm(y)
        y = self.activation(y)
        y = self.dropout(y)
        # 加入残差连接
        return y + self.residual(x)


class TCN(nn.Module):
    """
    TCN 时间序列预测模型
    """
    def __init__(self, input_dim, output_dim, num_layers, num_filters, kernel_size, dilation_base, dropout, prediction_horizon):
        """
        :param input_dim: 输入的特征数（每个时间步的变量数）
        :param output_dim: 输出的特征数（预测变量数）
        :param num_layers: TCN 的卷积层数
        :param num_filters: 每个卷积层的通道数
        :param kernel_size: 卷积核大小
        :param dilation_base: 膨胀因子基数
        :param dropout: Dropout 比例
        :param prediction_horizon: 预测的时间步长
        """
        super(TCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon

        # 创建 TCN 网络的多个膨胀卷积层
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            in_channels = input_dim if i == 0 else num_filters
            layers.append(
                TCNBlock(in_channels, num_filters, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)

        # 最终输出层，用于将 TCN 的特征映射到预测的时间序列
        self.output_layer = nn.Linear(num_filters, output_dim * prediction_horizon)

    def forward(self, x):
        """
        :param x: 输入的时间序列，形状 (batch_size, seq_length, input_dim)
        :return: 预测的时间序列，形状 (batch_size, prediction_horizon, output_dim)
        """
        x = x.transpose(1, 2)  # 转换为 (batch_size, input_dim, seq_length) 以适配 Conv1d
        features = self.tcn(x)  # 通过 TCN 提取特征
        features = features.mean(dim=-1)  # 对时间维度做全局池化
        predictions = self.output_layer(features)  # 映射到预测结果
        return predictions.view(x.shape[0], self.prediction_horizon, self.output_dim)  # 调整输出形状


# 测试模型
if __name__ == "__main__":
    batch_size = 64
    seq_length = 50
    input_dim = 3
    output_dim = 3
    prediction_horizon = 1
    model = TCN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=4,
        num_filters=32,
        kernel_size=3,
        dilation_base=2,
        dropout=0.2,
        prediction_horizon=prediction_horizon
    )

    # 创建一个随机输入序列
    x = torch.randn(batch_size, seq_length, input_dim)
    y = model(x)
    print(y.shape)  # 应输出: (batch_size, prediction_horizon, output_dim)
