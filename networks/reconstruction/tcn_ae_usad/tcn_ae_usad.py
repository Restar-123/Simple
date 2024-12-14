import math
from typing import Callable, Union, Tuple, List, Type

import torch
import torch.nn as nn

from .layers.tcn import TCN

from common.utils import set_device
import logging
import time
import numpy as np

class TCN_AE_USAD(nn.Module):
    def __init__(self, input_dimension: int,
                 device: int,
                 dilations: List[int] = (1, 2, 4, 8, 16),
                 nb_filters: Union[int, List[int]] = 3,
                 kernel_size: int = 10,
                 nb_stacks: int = 1,
                 padding: str = 'same',
                 dropout_rate: float = 0.00,
                 filters_conv1d: int = 8,
                 activation_conv1d: Union[str, Callable] = 'linear',
                 ):
        super().__init__()

        self.encoder = TCN(input_dimension, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=True, dropout_rate=dropout_rate,
                           return_sequences=True,activation="relu")

        self.decoder1 = TCN(filters_conv1d, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                            dilations=dilations, padding=padding, use_skip_connections=True, dropout_rate=dropout_rate,
                            return_sequences=True,activation="relu")

        self.decoder2 = TCN(filters_conv1d, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                            dilations=dilations, padding=padding, use_skip_connections=True, dropout_rate=dropout_rate,
                            return_sequences=True,activation="relu")
        self.device = set_device(device)
        self.to(self.device)

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        return w1.transpose(1, 2),w2.transpose(1, 2),w3.transpose(1, 2)

    def training_step(self, x, n):
        x = x.transpose(1, 2)
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((x - w1) ** 2) + (1 - 1 / n) * torch.mean((x - w3) ** 2)
        loss2 = 1 / n * torch.mean((x - w2) ** 2) + (1 - 1 / n) * torch.mean((x - w3) ** 2)
        return loss1,loss2

    def fit(self, train_loader, val_loader, epochs, lr, ):
        fit_usad(self,train_loader,epochs,lr)

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        alpha = 0.5
        beta = 1-alpha
        with torch.no_grad():
            for input in dataloader:
                input = input.to(self.device)
                w1,w2,w3 = self(input)
                loss = alpha * mse_func(input, w1) + beta * mse_func(input,w2)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps).mean(axis=(2,1))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores


def fit_usad(model, train_loader, epochs, lr, criterion=nn.MSELoss()):
    opt_func=torch.optim.Adam
    optimizer1 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder.parameters())
    )
    optimizer2 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder2.parameters())
    )
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        model.train()
        for input in train_loader:
            input = input.to(model.device)
            loss1,loss2 = model.training_step(input, epoch + 1)

            # 反向传播和优化
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss1, loss2 = model.training_step(input, epoch + 1)
            # 反向传播和优化
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # 累计损失
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        epoch_time = time.time() - epoch_start
        s = (
            f"[Epoch {epoch + 1}] "
            f"loss_1 = {epoch_loss1 / len(train_loader):.5f}, "
            f"loss_2 = {epoch_loss2 / len(train_loader):.5f}, "
        )
        s += f" [{epoch_time:.2f}s]"
        logging.info(s)

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")

if __name__ == '__main__':
    model = TCN_AE_USAD(
            input_dimension=3,
            device=0,
            dilations=(1, 2, 4, 8, 16),
            nb_filters=3,
            kernel_size=3,
            nb_stacks=1,
            padding='same',
            dropout_rate=0.00,
            filters_conv1d=3,
            activation_conv1d='relu',
        )
    input = torch.randn(64,50,3)
    input = input.to(model.device)
    output = model.training_step(input,1)
    print(output)