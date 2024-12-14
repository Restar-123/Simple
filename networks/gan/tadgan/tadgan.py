import itertools
from typing import Tuple, List, Iterator

import torch.nn as nn
from torch.nn import Parameter
from .layers.layers import *

from common.utils import set_device
import logging
import time
import numpy as np

class TADGAN(nn.Module):
    def __init__(self, input_size: int, window_size: int, device=0, latent_size: int = 20, enc_lstm_hidden_size: int = 100,
                 gen_lstm_hidden_size: int = 64, disc_conv_filters: int = 64, disc_conv_kernel_size: int = 5,
                 disc_z_hidden_size: int = 20, gen_dropout: float = 0.2, disc_dropout: float = 0.25,
                 disc_z_dropout: float = 0.2):
        super(TADGAN, self).__init__()

        self.encoder = TADGANEncoder(input_size, window_size, enc_lstm_hidden_size, latent_size)
        self.generator = TADGANGenerator(window_size, input_size, latent_size, gen_lstm_hidden_size, gen_dropout)
        self.discriminatorx = TADGANDiscriminatorX(input_size, window_size, disc_conv_filters, disc_conv_kernel_size,
                                                   disc_dropout)
        self.discriminatorz = TADGANDiscriminatorZ(latent_size, disc_z_hidden_size, disc_z_dropout)

        self.gan = GAN(self.generator, self.discriminatorx)
        self.inverse_gan = GAN(self.encoder, self.discriminatorz)

        self.latent_size = latent_size
        self.device = set_device(device)
        self.to(self.device)

    def grouped_parameters(self) -> Tuple[Iterator[Parameter], ...]:
        return (self.discriminatorz.parameters(), self.discriminatorx.parameters(),
                itertools.chain(self.encoder.parameters(), self.generator.parameters()))

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        real_z, real_x = inputs
        # (B, latent), (B, T, D)

        fake_x, real_x_score, fake_x_score = self.gan((real_z, real_x))
        # (B, T, D), (B, 1), (B, 1)

        fake_z, real_z_score, fake_z_score = self.inverse_gan((real_x, real_z))
        # (B, latent), (B, 1), (B, 1)

        reconstructed_x = self.generator(fake_z)
        # (B, T, D)

        return fake_z, fake_x, real_z_score, fake_z_score, real_x_score, fake_x_score, reconstructed_x

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_tadgan(self, train_loader, epochs, lr, criterion=nn.MSELoss())

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)
                real_x = input
                real_z = torch.randn(real_x.shape[0], self.latent_size, dtype=real_x.dtype, device=real_x.device)

                fake_z, fake_x, real_z_score, fake_z_score, real_x_score, fake_x_score, reconstructed_x = self(
                    (real_z, real_x))
                # 计算重构误差
                loss = torch.mean((reconstructed_x - input) ** 2, dim=(1, 2))  # (B,)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps)
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_tadgan(model, dataloader, epochs, lr, criterion=nn.MSELoss()):
    optimizerD = torch.optim.AdamW(model.gan.parameters(), lr, weight_decay=1e-5)
    optimizerG = torch.optim.AdamW(model.generator.parameters(), lr, weight_decay=1e-5)
    cross_entropy = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    train_start = time.time()
    torch.nn.MSELoss()
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
            real_x = input
            real_z = torch.randn(real_x.shape[0], model.latent_size, dtype=real_x.dtype, device=real_x.device)

            fake_z, fake_x, real_z_score, fake_z_score, real_x_score, fake_x_score, reconstructed_x = model(
                (real_z, real_x))

            # 1. 计算判别器损失 (Discriminator Loss)
            loss_real = cross_entropy(real_x_score, torch.ones_like(real_x_score))  # 判别真实数据的损失
            loss_fake = cross_entropy(fake_x_score, torch.zeros_like(fake_x_score))  # 判别生成数据的损失
            d_loss = loss_real + loss_fake

            # 更新判别器
            optimizerD.zero_grad()  # 清空梯度
            d_loss.backward()  # 计算梯度
            optimizerD.step()  # 更新判别器的参数

            # 2. 计算生成器损失 (Generator Loss)

            fake_z, fake_x, real_z_score, fake_z_score, real_x_score, fake_x_score, reconstructed_x = model((real_z, real_x))

            gen_loss = -torch.mean(fake_x_score)
            enc_loss = -torch.mean(fake_z_score)
            rec_loss = mse(fake_x, real_x)

            g_loss = gen_loss + enc_loss + rec_loss

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
