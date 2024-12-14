# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Tuple

import torch
import torch.nn as nn

from .layers.embed import DataEmbedding
from .layers.autocorrelation import AutoCorrelationLayer
from .layers.fourier_correlation import FourierBlock
from .layers.multi_wavelet_correlation import MultiWaveletTransform
from .layers.autoformer_encdec import Encoder, EncoderLayer, CustomLayerNorm, SeriesDecomp

from common.utils import set_device
import logging
import time
import numpy as np

class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(
            self,
            window_size: int,
            input_dim: int,
            device: 0,
            moving_avg: int=25,
            model_dim: int=128,
            dropout: float=0.1,
            num_heads: int=8,
            fcn_dim: int=128,
            activation: str='gelu',
            encoder_layers: int=3,
            version: str='fourier',
            mode_select: str='random',
            modes: int=32
        ) -> None:
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(FEDformer, self).__init__()
        self.seq_len = window_size

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.decomp = SeriesDecomp(moving_avg)
        self.enc_embedding = DataEmbedding(input_dim, model_dim, dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=model_dim, L=1, base='legendre')
        else:
            encoder_self_att = FourierBlock(in_channels=model_dim,
                                            out_channels=model_dim,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        model_dim, num_heads),
                    model_dim,
                    fcn_dim,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(encoder_layers)
            ],
            norm_layer=CustomLayerNorm(model_dim)
        )

        self.projection = nn.Linear(model_dim, input_dim, bias=True)

        self.device = set_device(device)
        self.to(self.device)


    def forward(self, x):
        x_enc = x
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

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
            loss = criterion(output, input)
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

def predict_prob_template(model, dataloader, window_labels=None):
    model.eval()
    mse_func = nn.MSELoss(reduction="none")
    loss_steps = []
    with torch.no_grad():
        for input in dataloader:
            input = input.to(model.device)
            output = model(input)
            loss = mse_func(input, output)
            loss_steps.append(loss.detach().cpu().numpy())
    anomaly_scores = np.concatenate(loss_steps).mean(axis=(2, 1))
    if window_labels is not None:
        anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
        return anomaly_scores, anomaly_label
    else:
        return anomaly_scores

