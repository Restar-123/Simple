from typing import List, Tuple, Iterator, Dict, Callable

import torch
import torch.nn.functional as F
import torch.nn as nn

from .layers.layers import GAN
from .layers.rnn import RNN

from common.utils import set_device
import logging
import time
import numpy as np


class MADGAN(GAN, nn.Module):
    def __init__(self, input_dim: int=3,  device =0, latent_dim: int = 3, generator_hidden_dims: List[int] = [100, 100, 100],
                 discriminator_hidden_dims: List[int] = [100]):
        generator = torch.nn.Sequential(
            RNN('lstm', 's2s', latent_dim, generator_hidden_dims),
            torch.nn.Linear(generator_hidden_dims[-1], input_dim),
            torch.nn.Sigmoid()  # For data to be in range [0, 1]
        )
        discriminator = torch.nn.Sequential(
            RNN('lstm', 's2s', input_dim, discriminator_hidden_dims),
            torch.nn.Linear(discriminator_hidden_dims[-1], 1)
        )

        super(MADGAN, self).__init__(generator, discriminator)

        self.latent_dim = latent_dim

        self.device = set_device(device)
        self.to(self.device)

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_beatgan(self, train_loader, epochs, lr, criterion)

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)
                fake_x, real_x_score, fake_x_score = self((input,input))
                # 计算重构误差
                loss = torch.mean((fake_x - input) ** 2, dim=(1, 2))  # (B,)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps)
        logging.info("mse:" + str(anomaly_scores.mean()))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_beatgan(model, dataloader, epochs, lr, criterion=nn.MSELoss()):
    optimizerD = torch.optim.AdamW(model.discriminator.parameters(), lr, weight_decay=1e-5)
    optimizerG = torch.optim.AdamW(model.generator.parameters(), lr, weight_decay=1e-5)
    cross_entropy = torch.nn.BCEWithLogitsLoss()
    cross_entropy2 = torch.nn.BCEWithLogitsLoss()
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss_d = 0.0  # discriminator 的损失
        epoch_loss_g = 0.0  # generator 的损失

        for input in dataloader:
            # 取出输入数据并移动到设备
            input = input.to(model.device)
            # 优化判别器
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            # 前向传播
            fake_x, real_x_score, fake_x_score = model((input,input))

            # 1. 计算判别器损失 (Discriminator Loss)
            loss_real = cross_entropy(real_x_score, torch.ones_like(real_x_score))  # 判别真实数据的损失
            loss_fake = cross_entropy(fake_x_score, torch.zeros_like(fake_x_score))  # 判别生成数据的损失
            d_loss = loss_real + loss_fake

            # 更新判别器
            optimizerD.zero_grad()  # 清空梯度
            d_loss.backward()  # 计算梯度
            optimizerD.step()  # 更新判别器的参数

            fake_x, real_x_score, fake_x_score = model((input,input))

            # 2. 计算生成器损失 (Generator Loss)
            g_loss_adv = -cross_entropy2(fake_x_score, torch.zeros_like(fake_x_score))  # 生成器对抗损失
            g_loss_recon = criterion(fake_x, input)  # 重构误差（MSE）
            g_loss = g_loss_adv + g_loss_recon

            # 更新生成器
            optimizerG.zero_grad()  # 清空梯度
            g_loss.backward()  # 计算梯度
            optimizerG.step()  # 更新生成器的参数

            # 累计损失
            epoch_loss_d += d_loss.item()
            epoch_loss_g += g_loss.item()

        epoch_time = time.time() - epoch_start
        s = (
            f"[Epoch {epoch + 1}] "
            f"d_loss = {epoch_loss_d / len(dataloader):.5f}, "
            f"g_loss = {epoch_loss_g / len(dataloader):.5f}, "
        )
        s += f" [{epoch_time:.2f}s]"
        logging.info(s)
        print(s)

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")

