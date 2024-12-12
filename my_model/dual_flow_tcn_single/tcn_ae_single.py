import math
from typing import Callable, Union, Tuple, List, Type

import torch
import torch.nn as nn
from torch.nn import functional as F
from .layers.tcn import TCN

from .layers import torch_utils
from common.utils import set_device
import logging
import time
import numpy as np

class TCN_AE(nn.Module):
    def __init__(self, input_dimension: int=3,
                 device: int=0,
                 dilations: List[int] = (1, 2, 4, 8),
                 nb_filters: Union[int, List[int]] = 20,
                 kernel_size: int = 9,
                 nb_stacks: int = 1,
                 padding: str = 'same',
                 dropout_rate: float = 0.00,
                 filters_conv1d: int = 20,
                 activation_conv1d: Union[str, Callable] = 'linear',
                 latent_sample_rate: int = 20,
                 pooler: Type[torch.nn.Module] = torch.nn.AvgPool1d,
                 use_skip_connections = True,
                 ):
        super().__init__()

        self.tcn_enc = TCN(input_dimension, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,)

        self.conv1d = torch.nn.Conv1d(nb_filters, filters_conv1d, kernel_size=1, padding=padding)

        # Ensure activation is non-inplace ReLU if it's a string
        if isinstance(activation_conv1d, str):
            activation_conv1d = torch_utils.activations[activation_conv1d]

        if isinstance(activation_conv1d, torch.nn.ReLU):
            activation_conv1d = torch.nn.ReLU(inplace=False)
        self.activation = activation_conv1d

        self.pooler = pooler(kernel_size=latent_sample_rate)

        self.tcn_dec = TCN(filters_conv1d, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=False, dropout_rate=dropout_rate,
                           )

        self.linear = torch.nn.Conv1d(nb_filters, input_dimension, kernel_size=1, padding=padding)

        self.device = set_device(device)
        self.to(self.device)

    def forward(self, x) -> torch.Tensor:
        """

        :param inputs: Tuple with single Tensor of shape (B, T, D)
        :return:
        """
        # Transpose the input to the required format (B, D, T)
        x = x.transpose(1, 2)

        # Put signal through TCN. Output-shape: (B, nb_filters, T)
        x = self.tcn_enc(x)
        # Now, adjust the number of channels...
        x = self.conv1d(x)
        x = self.activation(x)

        # Do some average (max) pooling to get a compressed representation of the time series
        # (e.g. a sequence of length 8)
        seq_len = x.shape[-1]
        x = self.pooler(x)
        # x = self.activation(x)

        # Now we should have a short sequence, which we will upsample again and
        # then try to reconstruct the original series
        x = F.interpolate(x, seq_len, mode='nearest')
        x = self.tcn_dec(x)
        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        x = self.linear(x)

        # Put output dimensions in the correct order again
        x = x.transpose(1, 2)

        return x


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
            # 反向传播和优化
            optimizer.zero_grad()
            loss = criterion(output, input)

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
    model = TCN_AE(
            input_dimension=3,
            device=0,
            dilations=(1, 2, 4, 8, 16),
            nb_filters=20,
            kernel_size=20,
            nb_stacks=1,
            padding='same',
            dropout_rate=0.00,
            filters_conv1d=8,
            activation_conv1d='linear',
            latent_sample_rate=42,
            pooler=torch.nn.AvgPool1d,
        )
    input = torch.randn(64,50,3)
    input = input.to(model.device)
    output = model(input)
    print(output.shape)