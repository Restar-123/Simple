# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library
from typing import Tuple

import torch.nn as nn
import torch
from .layers.embed import DataEmbedding
from .layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.autoformer_encdec import Encoder, EncoderLayer, CustomLayerNorm, SeriesDecomp
from common.utils import set_device
import logging
import time
import numpy as np

class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(
            self,
            input_dim: int,
            device = 0,
            moving_avg: int=25,
            model_dim: int=128,
            dropout: float=0.1,
            attention_factor: int=1,
            num_heads: int=8,
            fcn_dim: int=128,
            activation: str='gelu',
            encoder_layers: int=3,
        ) -> None:
        super(Autoformer, self).__init__()

        # Decomp
        kernel_size = moving_avg
        self.decomp = SeriesDecomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding(input_dim, model_dim, dropout, use_pos=False)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(attention_factor, attention_dropout=dropout,
                                        output_attention=False),
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
        # Decoder
        self.projection = nn.Linear(model_dim, input_dim, bias=True)
        self.device = set_device(device)
        self.to(self.device)

    def forward(self,x):
        x_enc = x
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_template(self, train_loader, epochs, lr, criterion)

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


def fit_template(model, dataloader, epochs, lr, criterion = nn.MSELoss()):
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        for input in dataloader:
            input = input.to(model.device)
            output = model(input)
            loss = criterion(output,input)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        s = (
            f"[Epoch {epoch + 1}] "
            f"loss = {epoch_loss/len(dataloader):.5f}, "
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



