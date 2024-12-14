from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from common.utils import set_device
import logging
import time
import numpy as np

from .layers.vae import DenseVAEEncoder, VAE
from .optim.loss import MaskedVAELoss

class Donut(nn.Module):
    def __init__(self, input_dim: int, device=0, hidden_dims: List[int] = [100, 100], latent_dim: int = 20,
                 mask_prob: float = 0.01):
        """
        Xu2018

        :param input_dim: Should be window_size * features
        :param hidden_dims:
        :param latent_dim:
        """
        super(Donut, self).__init__()

        self.latent_dim = latent_dim
        self.mask_prob = mask_prob

        encoder = DenseVAEEncoder(input_dim, hidden_dims, latent_dim)
        decoder = DenseVAEEncoder(latent_dim, hidden_dims[::-1], input_dim)
        self.vae = VAE(encoder, decoder, logvar_out=False)

        self.device = set_device(device)
        self.to(self.device)
        self.num_mc_samples=1024

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (B, T, D)
        x, = inputs
        B, T, D = x.shape

        if self.training:
            # Randomly mask some inputs
            mask = torch.empty_like(x)
            mask.bernoulli_(1 - self.mask_prob)
            x = x * mask
        else:
            mask = None

        # Run the VAE
        x = x.view(x.shape[0], -1)
        mean_z, std_z, mean_x, std_x, sample_z = self.vae(x, return_latent_sample=True)

        # Reshape the outputs
        mean_x = mean_x.view(B, T, D)
        std_x = std_x.view(B, T, D)

        return mean_z, std_z, mean_x, std_x, sample_z, mask

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_donut(self, train_loader, epochs, lr, criterion)

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)

                B, T, D = input.shape

                x_vae = input.view(input.shape[0], -1)
                res = self.vae(x_vae, return_latent_sample=False, num_samples=self.num_mc_samples)

                z_mu, z_std, x_dec_mean, x_dec_std = res
                # Reshape the outputs
                x_dec_mean = x_dec_mean.view(-1, B, T, D)
                x_dec_std = x_dec_std.view(-1, B, T, D)

                # Compute MC approximation of Log likelihood
                nll_output = torch.sum(F.gaussian_nll_loss(x_dec_mean[:, :, -1, :], input[:, -1, :].unsqueeze(0),
                                                           x_dec_std[:, :, -1, :] ** 2, reduction='none'), dim=(0, 2))
                nll_output /= self.num_mc_samples
                loss_steps.append(nll_output.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps)
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_donut(model, dataloader, epochs, lr, criterion ):
    # 分别为模型的不同模块创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)

    # 定义损失函数
    criterion = MaskedVAELoss()
    train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        model.train()

        for input in dataloader:
            # 将数据移到设备
            input = input.to(model.device)
            ouput = model((input,))

            loss = criterion(ouput,(input,))
            # 反向传播和优化
            optimizer.zero_grad()
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

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")