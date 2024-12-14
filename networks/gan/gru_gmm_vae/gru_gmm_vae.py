from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn

from .layers.rnn import RNN
from .layers.vae import DenseVAEEncoder, VAE
from .optim.loss import GMMVAELoss

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

class RNNVAECategoricalEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, rnn_type: str = 'lstm', rnn_hidden_dims: List[int] = [60], categories: int = 10,
                 bidirectional: bool = False, mode: str = 's2s'):
        super(RNNVAECategoricalEncoder, self).__init__()

        self.rnn = RNN(rnn_type, mode, input_dim, rnn_hidden_dims, bidirectional=bidirectional)
        out_hidden_size = 2 * rnn_hidden_dims[-1] if bidirectional else rnn_hidden_dims[-1]
        self.linear = torch.nn.Linear(out_hidden_size, categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        rnn_out = self.rnn(x)
        logits = self.linear(rnn_out)

        return logits


class GRU_GMM_VAE(nn.Module):
    def __init__(self, input_dim: int, device=0, gru_hidden_dims: List[int] = [60], latent_dim: int = 8, gmm_components: int = 2):
        """
        Guo2018 (more or less)

        :param input_dim:
        :param gru_hidden_dims:
        :param latent_dim:
        """
        super(GRU_GMM_VAE, self).__init__()

        self.latent_dim = latent_dim
        self.gmm_components = gmm_components

        self.encoder_rnn = RNN('gru', 's2s', input_dim, gru_hidden_dims)
        self.encoder_component = torch.nn.Linear(gru_hidden_dims[-1], gmm_components)

        encoder_normal = DenseVAEEncoder(gru_hidden_dims[-1] + gmm_components, gru_hidden_dims, latent_dim)
        decoder = RNNVAEGaussianEncoder(latent_dim, 'gru', gru_hidden_dims[::-1], input_dim, logvar_out=False)
        self.vae = VAE(encoder_normal, decoder, logvar_out=False)

        self.prior_means = torch.nn.Parameter(torch.rand(gmm_components, latent_dim))
        self.prior_std = torch.nn.Parameter(torch.rand(gmm_components, latent_dim))

        self.softplus = torch.nn.Softplus()

        self.device = set_device(device)
        self.to(self.device)

    def get_prior(self, batch_size: int, seq_len: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mean = self.prior_means.view(1, 1, *self.prior_means.shape)
        mean = mean.expand(seq_len, batch_size, -1, -1)

        std = self.prior_std.view(1, 1, *self.prior_std.shape)
        std = self.softplus(std)
        std = std.expand(seq_len, batch_size, -1, -1)

        return mean, std

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        # x: (T, B, D)
        x, = inputs
        T, B, D = x.shape

        means_z_prior, stds_z_prior = self.get_prior(B, T)

        # Run RNN on the input
        hidden = self.encoder_rnn(x)
        # Get the categorical distribution for the mixture components
        component_logits = self.encoder_component(hidden)

        # Run the VAE for each component
        means_z = []
        stds_z = []
        means_x = []
        stds_x = []
        one_hot = torch.zeros(self.gmm_components, dtype=torch.float, device=x.device)
        for k in range(self.gmm_components):
            one_hot[k] = 1
            mean_z, std_z, mean_x, std_x = self.vae(torch.cat([hidden, one_hot.expand(T, B, -1)], dim=-1))
            one_hot[k] = 0

            means_z.append(mean_z)
            stds_z.append(std_z)
            means_x.append(mean_x)
            stds_x.append(std_x)

        means_z = torch.stack(means_z, dim=-2)
        stds_z = torch.stack(stds_z, dim=-2)
        means_x = torch.stack(means_x, dim=-2)
        stds_x = torch.stack(stds_x, dim=-2)

        return means_z, stds_z, means_x, stds_x, means_z_prior, stds_z_prior, component_logits

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_gru_gmm_vae(self, train_loader, epochs, lr, criterion)

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)

                # 获取编码器的隐藏状态
                hidden = self.encoder_rnn(input)

                # 获取组件的 logits 和 OneHotCategorical 分布
                component_logits = self.encoder_component(hidden)
                component_dist = torch.distributions.OneHotCategorical(logits=component_logits)
                cat_sample = component_dist.sample()

                # 将分类样本与隐藏状态拼接
                x_cat = torch.cat([hidden, cat_sample], dim=-1)
                # 获取 VAE 的输出
                z_mean, z_std, x_mean, x_std = self.vae(x_cat, force_sample=True)
                # 计算重建误差 (MSE)
                reconstruction_error = torch.mean((input - x_mean) ** 2, dim=(2,1))
                # 计算 KL 散度
                kl_divergence = torch.sum(0.5 * (torch.exp(z_std) ** 2 + z_mean ** 2 - 1 - z_std), dim=(2,1))
                # 计算异常分数
                anomaly_score = reconstruction_error + kl_divergence

                loss_steps.append(anomaly_score.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps)
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_gru_gmm_vae(model, dataloader, epochs, lr, criterion ):
    # 分别为模型的不同模块创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)

    # 定义损失函数
    criterion = GMMVAELoss()

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
