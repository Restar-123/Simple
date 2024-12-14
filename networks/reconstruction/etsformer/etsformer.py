# Implementation derived from Time Series Library https://github.com/thuml/Time-Series-Library

import torch
import torch.nn as nn

from .layers.embed import DataEmbedding
from .layers.layer import EncoderLayer, Encoder, DecoderLayer, Decoder
from common.utils import set_device
import logging
import time
import numpy as np


class ETSformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2202.01381
    """

    def __init__(
            self,
            window_size: int,
            input_dim: int,
            device: 0,
            model_dim: int=128,
            dropout: float=0.1,
            num_heads: int=8,
            fcn_dim: int=128,
            encoder_layers: int=3,
            activation: str='gelu',
            top_k: int=5,
        ) -> None:
        super(ETSformer, self).__init__()
        decoder_layers = encoder_layers

        # Embedding
        self.enc_embedding = DataEmbedding(input_dim, model_dim, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    model_dim, num_heads, input_dim, window_size, window_size, top_k,
                    dim_feedforward=fcn_dim,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(encoder_layers)
            ]
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    model_dim, num_heads, input_dim, window_size,
                    dropout=dropout,
                ) for _ in range(decoder_layers)
            ],
        )

        self.device = set_device(device)
        self.to(self.device)


    def forward(self, x):
        x_enc = x
        res = self.enc_embedding(x_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds
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