import math
from typing import Callable, Union, Tuple, List, Type

import torch
import torch.nn as nn

from .layers.tcn import TCN
from .layers import torch_utils
from common.utils import set_device
import logging
import time
import numpy as np

class TCN_PRED(nn.Module):
    def __init__(self, input_dimension: int=3,
                 device: int=0,
                 dilations: List[int] = (1, 2, 4, 8),
                 nb_filters: Union[int, List[int]] = 5,
                 kernel_size: tuple = 3,
                 nb_stacks: int = 1,
                 padding: str = 'same',
                 dropout_rate: float = 0.00,
                 output_dim: int = 3,
                 activation_conv1d: Union[str, Callable] = 'linear',
                 next_steps: int = 1,
                 use_skip_connections = False,
                 ):
        super().__init__()

        self.tcn_enc = TCN(input_dimension, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                           n_steps=next_steps)

        self.conv1d = torch.nn.Conv1d(nb_filters, output_dim, kernel_size=1, padding=padding)

        if isinstance(activation_conv1d, str):
            activation_conv1d = torch_utils.activations[activation_conv1d]

        self.activation = activation_conv1d

        self.device = set_device(device)
        self.to(self.device)

    def forward(self, x) -> torch.Tensor:
        """
        :inputs: (B, T, D)
        :return: (B, n, D)
        """
        # Transpose the input to the required format (B, D, T)
        x = x.transpose(1, 2) # (B, D, T)

        x = self.tcn_enc(x)  # (B, nb_filters, n)
        x = self.conv1d(x)  # (B, D, n)
        x = self.activation(x)
        x = x.transpose(1, 2) #(B, n, D)

        return x

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_prediction(self, train_loader, epochs, lr, criterion)

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input,label in dataloader:
                input = input.to(self.device)
                label = label.to(self.device)
                pred = self(input)
                loss = mse_func(label, pred)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps).mean(axis=(2, 1))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_prediction(model, dataloader, epochs, lr, criterion=nn.MSELoss()):
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        for input,label in dataloader:
            input = input.to(model.device)
            label = label.to(model.device)

            pred = model(input)
            # 反向传播和优化
            optimizer.zero_grad()
            loss = criterion(pred, label)

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

if __name__ == '__main__':
    model  = TCN_PRED(
            input_dimension=3,
            device=0,
            dilations=(1, 2, 4, 8, 16),
            nb_filters=20,
            kernel_size=20,
            nb_stacks=1,
            padding='same',
            dropout_rate=0.00,
            output_dim=3,
            activation_conv1d='linear',
        )
    input = torch.randn(64,50,3).to(model.device)
    ouput = model(input)
    print(ouput.shape)

