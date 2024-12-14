from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVR
from .optim.loss import VAELoss
from .layers.rnn import RNN
from .layers.vae import VAE

from common.utils import set_device
import logging
import time
import numpy as np


class RNNVAEGaussianEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, rnn_type: str = 'lstm', rnn_hidden_dims: List[int] = [60], latent_dim: int = 10,
                 bidirectional: bool = False, mode: str = 's2s', logvar_out: bool = True):
        super(RNNVAEGaussianEncoder, self).__init__()

        self.logvar = logvar_out

        self.rnn = RNN(rnn_type, mode, input_dim, rnn_hidden_dims, bidirectional=bidirectional)
        out_hidden_size = 2 * rnn_hidden_dims[-1] if bidirectional else rnn_hidden_dims[-1]
        self.linear = torch.nn.Linear(out_hidden_size, 2 * latent_dim)
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        rnn_out = self.rnn(x)
        mean, std_or_logvar = self.linear(rnn_out).tensor_split(2, dim=-1)

        if not self.logvar:
            std_or_logvar = self.softplus(std_or_logvar)

        return mean, std_or_logvar


class LSTM_VAE(nn.Module):
    def __init__(self, input_dim: int, device=0, lstm_hidden_dims: List[int] = [60], latent_dim: int = 20):
        """
        Base LSTMVAE

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        """
        super(LSTM_VAE, self).__init__()

        self.latent_dim = latent_dim

        encoder = RNNVAEGaussianEncoder(input_dim, rnn_type='lstm', rnn_hidden_dims=lstm_hidden_dims,
                                        latent_dim=latent_dim, logvar_out=True)
        decoder = RNNVAEGaussianEncoder(latent_dim, rnn_type='lstm', rnn_hidden_dims=lstm_hidden_dims[::-1],
                                        latent_dim=input_dim, logvar_out=False)
        self.vae = VAE(encoder, decoder, logvar_out=True)

        self.device = set_device(device)
        self.to(self.device)

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (T, B, D)
        x, = inputs
        T, B, D = x.shape

        mean_z_prior, logvar_z_prior = self.get_prior(B, T)

        # Run the VAE
        mean_z, logvar_z, mean_x, std_x = self.vae(x)

        return mean_z, logvar_z, mean_x, std_x, mean_z_prior, logvar_z_prior


    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_gru_gmm_vae(self, train_loader, epochs, lr, criterion)
    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)

                res = self.vae(input, return_latent_sample=True, force_sample=True)

                z_mean, z_std, x_mean, x_std, z = res

                loss = torch.mean((input - x_mean) ** 2, dim=(2,1))  # (B, T)

                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps)
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_gru_gmm_vae(model, dataloader, epochs, lr, criterion):
    # 分别为模型的不同模块创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)

    # 定义损失函数
    criterion = VAELoss()

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




class LSTMVAESoelch(LSTM_VAE):
    def __init__(self, input_dim: int, device=0,lstm_hidden_dims: List[int] = [60], latent_dim: int = 20,
                 prior_hidden_dim: int = 40):
        """
        Sölch2016

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        :param prior_hidden_dim:
        """
        super(LSTMVAESoelch, self).__init__(input_dim)

        self.prior_hidden_dim = prior_hidden_dim

        self.prior_rnn = torch.nn.LSTMCell(latent_dim, prior_hidden_dim)
        self.prior_linear = torch.nn.Linear(prior_hidden_dim, latent_dim)
        self.device = set_device(device)
        self.to(self.device)

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mean_z_prior = []
        hidden = None
        # Init the first prior
        mean_z_prior.append(self.prior_linear(torch.zeros(batch_size, self.prior_hidden_dim,
                                                          dtype=self.prior_linear.weight.dtype,
                                                          device=self.prior_linear.weight.device)))
        for t in range(1, seq_len):
            hidden = self.prior_rnn(mean_z_prior[t - 1], hidden)
            mean_z_prior.append(self.prior_linear(hidden[0]))

        mean_z_prior = torch.stack(mean_z_prior, dim=0)

        return mean_z_prior, None


class LSTMVAEPark(LSTM_VAE):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60], latent_dim: int = 20,
                 noise_std: float = 0.1):
        """
        Park2018

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        :param noise_std:
        """
        super(LSTMVAEPark, self).__init__(input_dim, lstm_hidden_dims, latent_dim)

        self.noise_std = noise_std

        self.prior_means = torch.nn.Parameter(torch.zeros(2, latent_dim))

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        alphas = torch.linspace(0, 1, steps=seq_len, dtype=self.prior_means.dtype, device=self.prior_means.device)
        alphas = alphas.view(-1, 1, 1)
        mean_z_prior = alphas * self.prior_means[0].view(1, 1, -1) + (1 - alphas) * self.prior_means[1].view(1, 1, -1)
        mean_z_prior = mean_z_prior.expand(seq_len, batch_size, self.latent_dim)

        return mean_z_prior, None

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        x_orig, = inputs
        if self.training:
            # Corrupt input data
            x = torch.randn_like(x_orig)
            x *= self.noise_std
            x += x_orig
        else:
            x = x_orig

        return super(LSTMVAEPark, self).forward((x,))

