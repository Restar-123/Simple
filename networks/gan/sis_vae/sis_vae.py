from typing import List, Tuple

import torch
import torch.nn as nn

from .layers.mlp import MLP
from .layers.vae import DenseVAEEncoder
from .layers.vae import sample_normal
from .optim.loss import SISVAELossWithGeneratedPrior

from common.utils import set_device
import logging
import time
import numpy as np


class SIS_VAE(nn.Module):
    def __init__(self, input_dim: int, device=0,
                 rnn_hidden_dim: int = 200, latent_dim: int = 40,
                 x_hidden_dims: List[int] = [100], z_hidden_dims: List[int] = [100],
                 enc_hidden_dims: List[int] = [100], dec_hidden_dims: List[int] = [100],
                 prior_hidden_dims: List[int] = [100]):
        """
        Li2021, ist aber im Prinzip nur Chung2015 mit einem extra loss term

        :param input_dim:
        :param lstm_hidden_dims:
        :param latent_dim:
        """
        super(SIS_VAE, self).__init__()

        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.x_embed = MLP(input_dim, x_hidden_dims[:-1], x_hidden_dims[-1], activation=torch.nn.ReLU(),
                           activation_after_last_layer=True)
        self.z_embed = MLP(latent_dim, z_hidden_dims[:-1], z_hidden_dims[-1], activation=torch.nn.ReLU(),
                           activation_after_last_layer=True)

        self.encoder = DenseVAEEncoder(x_hidden_dims[-1] + rnn_hidden_dim, enc_hidden_dims, latent_dim)
        self.decoder = DenseVAEEncoder(z_hidden_dims[-1] + rnn_hidden_dim, dec_hidden_dims, input_dim)
        self.prior_decoder = DenseVAEEncoder(rnn_hidden_dim, prior_hidden_dims, latent_dim)

        self.rnn_cell = torch.nn.GRUCell(x_hidden_dims[-1] + z_hidden_dims[-1], rnn_hidden_dim)

        self.softplus = torch.nn.Softplus()

        self.device = set_device(device)
        self.to(self.device)

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (T, B, D)
        x, = inputs
        # T, B, D = x.shape

        # First compute an embedding for x
        # (T, B, hidden_x)
        x = self.x_embed(x)

        hidden = [torch.zeros(x.shape[1], self.rnn_hidden_dim, dtype=x.dtype, device=x.device)]
        z_mean = []
        z_std = []
        z_sample = []
        for t in range(x.shape[0]):
            z_mean_t, z_std_t = self.encoder(torch.cat([x[t], hidden[t]], dim=-1))
            z_sample_t = sample_normal(z_mean_t, z_std_t, log_var=False)
            z_sample_t = self.z_embed(z_sample_t)

            hidden.append(self.rnn_cell(torch.cat([x[t], z_sample_t], dim=-1), hidden[t]))
            z_mean.append(z_mean_t)
            z_std.append(z_std_t)
            z_sample.append(z_sample_t)

        hidden = torch.stack(hidden[:-1], dim=0)
        z_mean = torch.stack(z_mean, dim=0)
        z_std = self.softplus(torch.stack(z_std, dim=0))
        z_sample = torch.stack(z_sample, dim=0)

        prior_mean, prior_std = self.prior_decoder(hidden)
        prior_std = self.softplus(prior_std)

        x_mean, x_std = self.decoder(torch.cat([z_sample, hidden], dim=-1))
        x_std = self.softplus(x_std)

        return z_mean, z_std, x_mean, x_std, prior_mean, prior_std


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

                z_mean, z_std, x_mean, x_std, prior_mean, prior_std = res

                loss = torch.mean((input - x_mean) ** 2, dim=(2,1))  # (B, T)

                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps)
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_omni_anomaly(model, dataloader, epochs, lr, criterion=nn.MSELoss(), ):
    # 分别为模型的不同模块创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)

    # 定义损失函数
    criterion = SISVAELossWithGeneratedPrior()

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