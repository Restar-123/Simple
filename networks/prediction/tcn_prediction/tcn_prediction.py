from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn

from .layers import MLP
from .layers import SameCausalZeroPad1d

from common.utils import set_device
import logging
import time
import numpy as np

class TCN_S2S_Prediction(nn.Module):
    def __init__(self, input_dim: int, device=0, filters: Sequence[int] = (64, 64, 64, 64, 64),
                 kernel_sizes: Sequence[int] = (3, 3, 3, 3, 3), dilations: Sequence[int] = (1, 2, 4, 8, 16),
                 last_n_layers_to_cat: int = 3, activation=torch.nn.ReLU()):
        """
        He2019

        :param input_dim:
        :param filters:
        :param kernel_sizes:
        :param dilations:
        :param last_n_layers_to_cat:
        :param activation:
        """
        super(TCN_S2S_Prediction, self).__init__()

        assert len(filters) == len(kernel_sizes) == len(dilations)
        assert 0 < last_n_layers_to_cat < len(filters)

        self.last_n_layers_to_cat = last_n_layers_to_cat
        self.activation = activation

        filters = [input_dim] + list(filters)

        modules = []
        for in_channels, out_channels, kernel_size, dilation in zip(filters[:-1], filters[1:], kernel_sizes, dilations):
            padding = SameCausalZeroPad1d(kernel_size, dilation=dilation)
            conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            modules.append(torch.nn.Sequential(padding, conv))
        self.conv_layers = torch.nn.ModuleList(modules)

        # Final 1x1 conv to retrieve the output
        self.final_conv = torch.nn.Conv1d(sum(filters[-last_n_layers_to_cat:]), input_dim, 1)
        self.device = set_device(device)
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = inputs
        # x: (B, D, T)
        x = x.transpose(1, 2)

        outputs = []
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = self.activation(x)

            if i >= len(self.conv_layers) - self.last_n_layers_to_cat:
                outputs.append(x)

        x_cat = torch.cat(outputs, dim=1)

        x_pred = self.final_conv(x_cat)

        return x_pred.transpose(1, 2)
    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_recon(self, train_loader, epochs, lr, criterion)

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


class TCNPrediction(nn.Module):
    def __init__(self, input_dim: int, window_size: int, device = 0, filters: Sequence[int] = (32, 32),
                 kernel_sizes: Sequence[int] = (3, 3), linear_hidden_layers: Sequence[int] = (50,),
                 activation: Union[Callable, str] = torch.nn.ReLU(), prediction_horizon: int = 1):
        """
        DeepAnT aka TCN prediction (Munir2018)
        :param input_dim:
        :param filters:
        :param kernel_sizes:
        :param linear_hidden_layers:
        :param activation:
        :param prediction_horizon:
        """
        super(TCNPrediction, self).__init__()

        assert len(filters) == len(kernel_sizes)

        self.activation = activation
        self.prediction_horizon = prediction_horizon
        self.pooler = torch.nn.MaxPool1d(2)

        filters = [input_dim] + list(filters)

        modules = []
        for in_channels, out_channels, kernel_size in zip(filters[:-1], filters[1:], kernel_sizes):
            conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
            modules.append(conv)
        self.conv_layers = torch.nn.ModuleList(modules)

        final_output_size = filters[-1] * int(window_size * 0.5**(len(filters) - 1))
        self.mlp = MLP(final_output_size, list(linear_hidden_layers), prediction_horizon * input_dim, activation)
        self.device = set_device(device)
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x= inputs
        x = x.transpose(1, 2)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.activation(x)
            x = self.pooler(x)

        # Flatten x
        x = x.view(x.shape[0], -1)

        # x_pred: (B, horizon * D)
        x_pred = self.mlp(x)
        # x_pred: (B, horizon, D)
        x_pred = x_pred.view(x_pred.shape[0], self.prediction_horizon, -1)
        return x_pred

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



def fit_recon(model, dataloader, epochs, lr, criterion = nn.MSELoss()):
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