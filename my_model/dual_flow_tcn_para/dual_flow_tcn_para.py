import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_device

from .layers.modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
)
from .tcn_ae_para import TCN_AE
from .tcn_pred_para import TCN_PRED


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

class Dual_Flow_TCN(nn.Module):
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
        super(Dual_Flow_TCN, self).__init__()

        window_size = window_size - next_steps
        self.n_features = n_features
        self.conv = ConvLayer(n_features, kernel_size)
        self.lsk_block = LSKblock1D(3)

        self.pred_model = TCN_PRED(next_steps=next_steps,dilations=pre_dilations,nb_filters=pre_nb_filters,kernel_size=pre_kernel_size,
                                   nb_stacks=pre_nb_stacks,padding=pre_padding,dropout_rate=pre_dropout_rate,activation_conv1d=pre_activation_conv1d,
                                   use_skip_connections=pre_use_skip_connections)
        self.recon_model = TCN_AE(dilations = ae_dilations, nb_filters= ae_nb_filters, kernel_size=ae_kernel_size,nb_stacks=ae_nb_stacks,
                                  padding=ae_padding,dropout_rate=ae_dropout,filters_conv1d=ae_filters_conv1d,activation_conv1d=ae_activation_conv1d,
                                  latent_sample_rate=ae_latent_sample_rate,use_skip_connections=ae_use_skip_connections)

        self.device = set_device(device)
        self.to(self.device)


    def forward(self, x):
        lsk_x = self.lsk_block(x)
        # convx = self.conv(x)
        # h_feat = self.feature_gat(x)
        # h_temp = self.temporal_gat(x)

        # h_end = x + h_feat + h_temp

        h_end = x + lsk_x
        pred = self.pred_model(h_end)
        recon = self.recon_model(h_end)

        return pred, recon

    def fit(self, train_loader, val_loader=None, epochs=20, lr=0.0001, criterion=nn.MSELoss()):
        fit_mtad_gat(self, train_loader,val_loader, epochs, lr, criterion=nn.MSELoss())

    def predict_prob(self, data_loader, window_labels=None):
        self.to(self.device)
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred,recon = self(x)
                window = torch.cat((x,y),dim=1)
                pred_window = torch.cat((pred,recon),dim=1)
                loss = mse_func(window, pred_window)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps).mean(axis=(2, 1))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_mtad_gat(model, train_loader, val_loader=None, epochs=20, lr=0.0001, criterion=nn.MSELoss()):
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_pred_loss = 0.0
        train_recon_loss = 0.0
        train_epoch_loss = 0.0
        for input,label in train_loader:
            input = input.to(model.device)
            label = label.to(model.device)
            pred,recon = model(input)

            recon_loss = criterion(recon, input)
            pred_loss = criterion(pred,label)
            loss = 0.5 * recon_loss + 0.5 * pred_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            train_epoch_loss += loss.item()
            train_pred_loss += pred_loss.item()
            train_recon_loss += recon_loss.item()

        if(val_loader is not None):
            model.eval()
            val_pred_loss = 0.0
            val_recon_loss = 0.0
            val_epoch_loss = 0.0
            with torch.no_grad():
                for input, label in val_loader:
                    input = input.to(model.device)
                    label = label.to(model.device)
                    pred, recon = model(input)

                    recon_loss = criterion(recon, input)
                    pred_loss = criterion(pred, label)
                    loss = 0.6 * recon_loss + 0.4 * pred_loss

                    # 累计损失
                    val_epoch_loss += loss.item()
                    val_pred_loss += pred_loss.item()
                    val_recon_loss += recon_loss.item()
            s = (f"[valid: ] "
                 f"[Epoch {epoch + 1}] "
                 f"pred_loss = {val_pred_loss / len(val_loader):.5f}, "
                 f"recon_loss = {val_recon_loss / len(val_loader):.5f}, "
                 f"loss = {val_epoch_loss / len(val_loader):.5f} "
                 )
            logging.info(s)
        epoch_time = time.time() - epoch_start
        s = (
            f"[train: ] "
            f"[Epoch {epoch + 1}] "
            f"pred_loss = {train_pred_loss / len(train_loader):.5f}, "
            f"recon_loss = {train_recon_loss / len(train_loader):.5f}, "
            f"loss = {train_epoch_loss / len(train_loader):.5f} "
        )
        s += f"  [{epoch_time:.2f}s]"
        logging.info(s)


    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")

if __name__ == '__main__':

    model = Dual_Flow_TCN(

        )
    input = torch.randn(64,49,3)
    label = torch.randn(64,1,3)
    pred,recon = model(input)
    print(pred.shape)
    print(recon.shape)
