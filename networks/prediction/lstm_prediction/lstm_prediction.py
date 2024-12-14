from typing import List, Union, Callable, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import RNN, MLP
from common.utils import set_device
import logging
import time
import numpy as np

class LSTMPrediction(nn.Module):
    def __init__(self, input_dim: int, device:0, lstm_hidden_dims: List[int] = [30, 20], linear_hidden_layers: List[int] = [],
                 linear_activation: Union[Callable, str] = torch.nn.ELU(), prediction_horizon: int = 3):
        """
        LSTM prediction (Malhotra2015)
        :param input_dim:
        :param lstm_hidden_dims:
        :param linear_hidden_layers:
        :param linear_activation:
        :param prediction_horizon:
        """
        super(LSTMPrediction, self).__init__()

        self.prediction_horizon = prediction_horizon

        self.lstm = RNN('lstm', 's2fh', input_dim, lstm_hidden_dims)
        self.mlp = MLP(lstm_hidden_dims[-1], linear_hidden_layers, prediction_horizon * input_dim, linear_activation)
        self.device = set_device(device)
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Reshape inputs to (T, B, D) for LSTM
        x = inputs.transpose(0, 1)

        # LSTM forward pass, get the last hidden state
        hidden = self.lstm(x)  # hidden: (B, hidden_dims)

        # MLP for prediction
        x_pred = self.mlp(hidden)  # x_pred: (B, horizon * D)

        # Reshape to (B, horizon, D)
        x_pred = x_pred.view(x_pred.shape[0], self.prediction_horizon, -1)

        # Output as (B, horizon, D)
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





class LSTM_S2S_Prediction(nn.Module):
    def __init__(self, input_dim: int, device:0, lstm_hidden_dims: List[int] = [30, 20], linear_hidden_layers: List[int] = [],
                 linear_activation: Union[Callable, str] = torch.nn.ELU(), dropout: float = 0.0):
        """
        LSTM prediction (Filonov2016)
        :param input_dim:
        :param lstm_hidden_dims:
        :param linear_hidden_layers:
        :param linear_activation:
        """
        super(LSTM_S2S_Prediction, self).__init__()

        self.lstm = RNN('lstm', 's2s', input_dim, lstm_hidden_dims, dropout=dropout)
        self.mlp = MLP(lstm_hidden_dims[-1], linear_hidden_layers, input_dim, linear_activation)
        self.device = set_device(device)
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # x: (T, B, D)
        x = inputs

        # hidden: (T, B, hidden_dims)
        hidden = self.lstm(x)
        # x_pred: (T, B, D)
        x_pred = self.mlp(hidden)

        return x_pred

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_recon(self, train_loader, epochs, lr, criterion=nn.MSELoss())

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