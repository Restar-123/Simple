import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from .layers.rnn import RNN
from .layers.vae import DenseVAEEncoder
from .layers.planar_nf import PlanarFlow
from .optim.loss import OmniAnomalyLoss

from common.utils import set_device
import logging
import time
import numpy as np


class KalmanFilter(torch.nn.Module):
    def __init__(self, state_dim: int, observation_dim: int):
        super(KalmanFilter, self).__init__()

        self.state_dim = state_dim
        self.observation_dim = observation_dim

        self.F = torch.nn.Parameter(torch.eye(state_dim, dtype=torch.float))
        self.state_cov = torch.nn.Parameter(torch.ones(state_dim, dtype=torch.float))

        H = torch.zeros(observation_dim, state_dim, dtype=torch.float)
        H.diagonal().add_(1)
        self.H = torch.nn.Parameter(H)
        self.obs_cov = torch.nn.Parameter(torch.ones(observation_dim, dtype=torch.float))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (T, L, B, D)
        T, B, D = observations.shape

        assert D == self.observation_dim

        s_cov = F.softplus(self.state_cov)
        obs_cov = F.softplus(self.obs_cov)

        log_likelihood = 0
        state_mean = torch.zeros(B, self.state_dim, dtype=observations.dtype, device=observations.device)
        state_cov = torch.eye(self.state_dim, dtype=observations.dtype, device=observations.device)

        for t in range(T):
            observation = observations[t]
            state_upd_mean = state_mean @ self.F.T
            state_update_cov = self.F @ state_cov @ self.F.T + s_cov.diag_embed()

            innovation = observation - state_upd_mean @ self.H.T
            innovation_cov = self.H @ state_update_cov @ self.H.T + obs_cov.diag_embed()

            expanded_innovation_cov = innovation_cov.view(1, self.observation_dim, self.observation_dim)
            expanded_innovation_cov = expanded_innovation_cov.expand(B, self.observation_dim, self.observation_dim)
            solve_mean = torch.linalg.solve(expanded_innovation_cov, innovation)
            solve_cov = torch.linalg.solve(innovation_cov, self.H)

            PH = state_update_cov @ self.H.T

            state_mean = state_upd_mean + solve_mean @ PH.T
            state_cov = state_update_cov - PH @ solve_cov @ state_update_cov

            log_likelihood = log_likelihood - 0.5 * (torch.inner(innovation, solve_mean) + torch.logdet(innovation_cov)
                                                     + self.observation_dim * math.log(2 * math.pi))

        return log_likelihood


class OmniAnomaly(nn.Module):
    def __init__(self, input_dim: int, device=0, latent_dim: int = 3, rnn_hidden_dims: Sequence[int] = (500,),
                 dense_hidden_dims: Sequence[int] = (500, 500), nf_layers: int = 20):
        super(OmniAnomaly, self).__init__()

        self.latent_dim = latent_dim

        self.prior = KalmanFilter(latent_dim, latent_dim)

        self.enc_rnn = RNN('gru', 's2s', input_dim, rnn_hidden_dims)
        self.encoder_vae = DenseVAEEncoder(rnn_hidden_dims[-1] + latent_dim, dense_hidden_dims, latent_dim)
        self.latent_nf = PlanarFlow(latent_dim, num_layers=nf_layers)

        self.decoder_rnn = RNN('gru', 's2s', latent_dim, rnn_hidden_dims)
        self.decoder_vae = DenseVAEEncoder(rnn_hidden_dims[-1], dense_hidden_dims, input_dim)

        self.device = set_device(device)
        self.to(self.device)

    def forward(self, inputs: Tuple[torch.Tensor], num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                  torch.Tensor, torch.Tensor,
                                                                                  torch.Tensor, torch.nn.Module,
                                                                                  torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        x, = inputs
        T, B, D = x.shape

        # Use RNN to encode the input
        hidden = self.enc_rnn(x)
        # Add sample dimension
        hidden = hidden.unsqueeze(1).expand(T, num_samples, B, hidden.shape[-1])

        normal_sample = torch.randn((T+1, num_samples, B, self.latent_dim), dtype=x.dtype, device=x.device)
        z_sample = [normal_sample[0]]
        z_mean, z_std = [], []
        for t in range(T):
            z_t_mean, z_t_std = self.encoder_vae(torch.cat([hidden[t], z_sample[t]], dim=-1))
            z_mean.append(z_t_mean)
            z_std.append(z_t_std)
            # From the paper it might seem that the normalizing flow is used during the sequential computation of z,
            # but in the code this is not the case. We choose to implement it like in the code
            z_sample.append(z_t_std * normal_sample[t+1] + z_t_mean)
        z_mean = torch.stack(z_mean, dim=0)
        z_std = torch.stack(z_std, dim=0)
        z_sample = torch.stack(z_sample[1:], dim=0)

        # Transform samples using a planar normalizing flow
        z_sample_transformed, z_log_det = self.latent_nf(z_sample)

        # Decode. Collapse sample and batch dimension for RNN
        z_sample_transformed = z_sample_transformed.view(T, -1, self.latent_dim)
        dec_hidden = self.decoder_rnn(z_sample_transformed)
        # Restore sample dimension
        dec_hidden = dec_hidden.view(T, num_samples, B, dec_hidden.shape[-1])
        x_rec_mean, x_rec_std = self.decoder_vae(dec_hidden)

        return z_std, z_mean, z_sample, z_sample_transformed, z_log_det, self.prior, x_rec_mean, x_rec_std

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_omni_anomaly(self, train_loader, epochs, lr, criterion)
    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)

                res = self((input,))

                z_std, z_mean, z_sample, z_sample_transformed, z_log_det, prior, x_rec_mean, x_rec_std = res

                loss = torch.mean((input - x_rec_mean.squeeze(1)) ** 2, dim=(2,1))  # (B, T)

                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps)
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_omni_anomaly(model, dataloader, epochs, lr, criterion):
    # 分别为模型的不同模块创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)

    # 定义损失函数
    criterion = OmniAnomalyLoss()

    train_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        epoch_loss = 0.0

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
        print(s)

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")