import itertools
from inspect import Parameter
from typing import List, Tuple, Iterator, Dict, Callable,Union

import torch
import torch.nn as nn
from .layers.rnn import RNN
from .layers.vae import VAE
from .optim.loss import VAELoss,GANGeneratorLoss,GANDiscriminatorLoss

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


class LSTMVAEGANDecoder(torch.nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60], latent_dim: int = 10):
        super(LSTMVAEGANDecoder, self).__init__()

        self.rnn = RNN('lstm', 's2s', latent_dim, lstm_hidden_dims)
        self.linear_mean = torch.nn.Linear(lstm_hidden_dims[-1], input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        rnn_out = self.rnn(x)

        mean = self.linear_mean(rnn_out)

        # The paper always assumes std of one
        return mean


class LSTMVAEGANDiscriminator(torch.nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_dims: List[int] = [60]):
        super(LSTMVAEGANDiscriminator, self).__init__()

        self.rnn = RNN('lstm', 's2s', input_dim, lstm_hidden_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (T, B, D)
        rnn_out = self.rnn(x)

        return rnn_out, torch.ones_like(rnn_out)


class LSTM_VAE_GAN(nn.Module):
    def __init__(self, input_dim: int, device=0, lstm_hidden_dims: List[int] = [60], latent_dim: int = 10):
        super(LSTM_VAE_GAN, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = RNNVAEGaussianEncoder(input_dim, rnn_type='lstm', rnn_hidden_dims=lstm_hidden_dims,
                                             latent_dim=latent_dim)
        self.decoder = LSTMVAEGANDecoder(input_dim, lstm_hidden_dims, latent_dim)
        self.discriminator = LSTMVAEGANDiscriminator(input_dim, lstm_hidden_dims)
        self.classifier = torch.nn.Linear(lstm_hidden_dims[-1], 1)

        self.vae = VAE(self.encoder, torch.nn.Sequential(self.decoder, self.discriminator))

        self.device = set_device(device)
        self.to(self.device)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        # z: random sample (T, B, latent_dim)
        # x: (T, B, D)
        z, x = inputs

        z_mu, z_log_var, x_rec, _ = self.vae(x)
        x, _ = self.discriminator(x)
        x_gen, _ = self.discriminator(self.decoder(z))

        x_score = self.classifier(x)
        x_rec_score = self.classifier(x_rec)
        x_gen_score = self.classifier(x_gen)

        return z_mu, z_log_var, x, x_rec, x_score, x_rec_score, x_gen_score

    def fit(self, train_loader, val_loader,epochs, lr, criterion=nn.MSELoss()):
        fit_lstm_vae_gan(self, train_loader, epochs, lr, criterion)
    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)
                disc_score, _ = self.discriminator(input)
                z_mean, z_std = self.encoder(input)
                pred = self.decoder(z_mean)
                loss = mse_func(input,pred)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps).mean(axis=(2, 1))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_lstm_vae_gan(model, dataloader, epochs, lr, criterion=nn.MSELoss(),):
    # 分别为模型的不同模块创建优化器
    enc_opt = torch.optim.AdamW(model.encoder.parameters(), lr, weight_decay=1e-5)
    dec_opt = torch.optim.AdamW(model.decoder.parameters(), lr, weight_decay=1e-5)
    dis_opt = torch.optim.AdamW(model.discriminator.parameters(), lr, weight_decay=1e-5)
    clf_opt = torch.optim.AdamW(model.classifier.parameters(), lr, weight_decay=1e-5)

    # 定义损失函数
    vae_loss_fn = VAELoss()
    gen_loss_fn = GANGeneratorLoss()
    dis_loss_fn = GANDiscriminatorLoss()

    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        # 初始化本轮的累计损失
        epoch_loss_enc = 0.0  # VAE 部分的损失
        epoch_loss_dec = 0.0  # 生成器损失
        epoch_loss_dis = 0.0  # 判别器损失

        for input in dataloader:
            # 将数据移到设备
            input = input.to(model.device)

            real_x = input
            real_target = input

            x_target, _ = model.discriminator(real_target)

            # Train encoder
            z_mu, z_log_var, x_rec, x_rec_std = model.vae(real_x)
            l_vae = vae_loss_fn((z_mu, z_log_var, x_rec, x_rec_std), (x_target,))

            enc_opt.zero_grad(True)
            l_vae.backward(inputs=list(model.encoder.parameters()))
            enc_opt.step()
            enc_opt.zero_grad(True)
            epoch_loss_enc += l_vae.item()

            # Train decoder
            z_mu, z_log_var, x_rec, x_rec_std = model.vae(real_x)
            l_vae = vae_loss_fn((z_mu, z_log_var, x_rec, x_rec_std), (x_target,))

            x_rec_score = model.classifier(x_rec)
            l_disc = gen_loss_fn((None, None, x_rec_score), (None,))

            l = l_vae + l_disc

            dec_opt.zero_grad(True)
            l.backward(inputs=list(model.decoder.parameters()))
            dec_opt.step()
            dec_opt.zero_grad(True)
            epoch_loss_dec += l.item()

            # Train discriminator
            # Generate a random vector for z
            real_z = torch.randn(real_x.shape[0], real_x.shape[1], model.latent_dim, dtype=real_x.dtype,
                                 device=real_x.device)
            # (T, B, latent)

            z_mu, z_log_var, x, x_rec, x_score, x_rec_score, x_gen_score = model((real_z, real_x))

            l_disc = dis_loss_fn((x_rec, x_score, x_rec_score), (None,))

            dis_opt.zero_grad(True)
            l_disc.backward(
                inputs=list(itertools.chain(model.discriminator.parameters(), model.classifier.parameters())))
            dis_opt.step()
            dis_opt.zero_grad(True)
            epoch_loss_dis += l_disc.item()

        epoch_time = time.time() - epoch_start
        s = (
            f"[Epoch {epoch + 1}] "
            f"vae_loss = {epoch_loss_enc / len(dataloader):.5f}, "
            f"dis_loss = {epoch_loss_dec / len(dataloader):.5f}, "
            f"gen_loss = {epoch_loss_dis / len(dataloader):.5f}, "
            f" [{epoch_time:.2f}s]"
        )
        logging.info(s)
        print(s)

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")