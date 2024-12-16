import os
import math
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_device
from .layers.tcn import TCN
from torch.nn.parameter import Parameter

from torch.nn import functional as F

from .layers.modules import (
    ConvLayer,
)


class LSKblock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度卷积操作，局部特征提取
        self.conv0 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # 大感受野深度卷积
        self.conv_spatial = nn.Conv1d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        # 降维的逐点卷积
        self.conv1 = nn.Conv1d(dim, dim // 2, kernel_size=1)
        self.conv2 = nn.Conv1d(dim, dim // 2, kernel_size=1)
        # 注意力权重计算（均值和最大值特征融合）
        self.conv_squeeze = nn.Conv1d(2, 2, kernel_size=7, padding=3)
        # 恢复通道维度
        self.conv = nn.Conv1d(dim // 2, dim, kernel_size=1)

    def forward(self, x):
        """
        输入: x -> (N, C, T)
        输出: x -> (N, C, T)
        """
        x = x.transpose(1, 2)
        attn1 = self.conv0(x)  # 局部特征提取 (N, C, T)
        attn2 = self.conv_spatial(attn1)  # 全局特征提取 (N, C, T)

        attn1 = self.conv1(attn1)  # 降维 (N, C//2, T)
        attn2 = self.conv2(attn2)  # 降维 (N, C//2, T)

        attn = torch.cat([attn1, attn2], dim=1)  # 拼接 (N, C, T)

        # 通道维度上的平均池化和最大池化
        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # (N, 1, T)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # (N, 1, T)

        # 融合注意力特征
        agg = torch.cat([avg_attn, max_attn], dim=1)  # (N, 2, T)
        sig = self.conv_squeeze(agg).sigmoid()  # 生成注意力权重 (N, 2, T)

        # 动态加权
        attn = attn1 * sig[:, 0, :].unsqueeze(1) + attn2 * sig[:, 1, :].unsqueeze(1)  # (N, C//2, T)
        attn = self.conv(attn)  # 恢复通道维度 (N, C, T)

        y = x * attn
        y = y.transpose(1, 2)
        return y  # 输出 (N, C, T)


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )

class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
        return y

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class AAMP_NO_PRED_LSK_MEMORY(nn.Module):
    def __init__(
        self,
        n_features=3,
        device=0,

        window_size=50,
        kernel_size=3,
        next_steps=1,

        ae_dilations=(1, 2, 4, 8),
        ae_nb_filters=20,
        ae_kernel_size=(3,9),
        ae_nb_stacks=1,
        ae_padding="same",
        ae_dropout=0.0,
        ae_filters_conv1d=20,
        ae_activation_conv1d='linear',
        ae_latent_sample_rate=20,
        ae_use_skip_connections=True,

        pre_dilations=(1, 2, 4, 8),
        pre_nb_filters=20,
        pre_kernel_size=(3,9),
        pre_nb_stacks=1,
        pre_padding='same',
        pre_dropout_rate=0.0,
        pre_activation_conv1d='linear',
        pre_use_skip_connections=True,
    ):
        super(AAMP_NO_PRED_LSK_MEMORY, self).__init__()

        window_size = window_size - next_steps
        self.n_features = n_features
        self.conv = ConvLayer(n_features, kernel_size)
        self.lsk_block = LSKblock1D(3)

        self.encoder = TCN(3, nb_filters=ae_nb_filters, kernel_size=ae_kernel_size, nb_stacks=ae_nb_stacks,
                           dilations=ae_dilations, padding=ae_padding, use_skip_connections=ae_use_skip_connections, dropout_rate=ae_dropout,
                           n_steps=0)
        self.memory = MemModule(mem_dim=100, fea_dim=ae_nb_filters, shrink_thres=0.0025)

        self.recon = TCN(ae_nb_filters, nb_filters=ae_nb_filters, kernel_size=ae_kernel_size, nb_stacks=ae_nb_stacks,
                           dilations=ae_dilations, padding=ae_padding, use_skip_connections=ae_use_skip_connections, dropout_rate=ae_dropout,
                           n_steps=0)

        self.activation = torch.nn.LeakyReLU(negative_slope=0.2)
        self.linear = torch.nn.Conv1d(ae_nb_filters, 3, kernel_size=1, padding=ae_padding)
        self.device = set_device(device)
        self.to(self.device)


    def forward(self, x):
        # lsk_x = self.lsk_block(x)

        x = x
        x = x.transpose(1, 2)
        z = self.encoder(x)
        z = self.activation(z)

        # z = self.memory(z)

        recon = self.recon(z)
        recon = self.linear(recon)
        recon = recon.transpose(1, 2)

        return recon

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_template(self, train_loader, epochs, lr, criterion=nn.MSELoss())

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)
                output = self(input)
                loss = mse_func(input, output)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps).mean(axis=(2, 1))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_template(model, dataloader, epochs, lr, criterion=nn.MSELoss()):
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        for input in dataloader:
            input = input.to(model.device)
            output = model(input)
            # 反向传播和优化
            optimizer.zero_grad()
            loss = criterion(output, input)

            loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        s = (
            f"[Epoch {epoch + 1}] "
            f"loss = {epoch_loss / len(dataloader):.5f}, "
        )
        s += f" [{epoch_time:.2f}s]"
        logging.info(s)
        print(s)

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")