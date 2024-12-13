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
from torch.nn.parameter import Parameter

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )

class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
        return y

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class TCN_AE(nn.Module):
    def __init__(self, input_dimension: int = 3,
                 device: int = 0,
                 dilations: List[int] = (1, 2, 4, 8),
                 nb_filters: Union[int, List[int]] = 20,
                 kernel_size: tuple = (3,9),
                 nb_stacks: int = 1,
                 padding: str = 'same',
                 dropout_rate: float = 0.00,
                 filters_conv1d: int = 20,
                 activation_conv1d: Union[str, Callable] = 'linear',
                 latent_sample_rate: int = 42,
                 use_skip_connections = False,
                 pooler: Type[torch.nn.Module] = torch.nn.AvgPool1d):
        super().__init__()

        self.tcn_enc = TCN(input_dimension, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                           n_steps=0)
        # 逐点卷积，调整通道数
        self.conv1d = torch.nn.Conv1d(nb_filters, filters_conv1d, kernel_size=1, padding=padding)
        self.memory = MemModule(mem_dim=30, fea_dim=filters_conv1d, shrink_thres=0.0025)

        # Ensure activation is non-inplace ReLU if it's a string
        if isinstance(activation_conv1d, str):
            activation_conv1d = torch_utils.activations[activation_conv1d]
        self.activation = activation_conv1d

        self.tcn_dec = TCN(filters_conv1d, nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks,
                           dilations=dilations, padding=padding, use_skip_connections=use_skip_connections, dropout_rate=dropout_rate,
                           n_steps=0)


        # 逐点卷积，调整通道数
        self.linear = torch.nn.Conv1d(nb_filters, input_dimension, kernel_size=1, padding=padding)

    def forward(self, x) -> torch.Tensor:
        """
        :param inputs: (B, T, D)
        :return:       (B, T, D)
        """
        x = x.transpose(1, 2)   #  (B, D, T)

        x = self.tcn_enc(x)   # (B, nb_filters, T)
        x = self.conv1d(x)    # (B, filters_conv1d, T)
        x = self.memory(x)
        x = self.tcn_dec(x)  # (B, nb_filters, T)
        x = self.linear(x)  #  (B, D, T)

        x = x.transpose(1, 2)  #  (B, T, D)
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
            nb_filters=3,
            kernel_size=(3,7),
            nb_stacks=1,
            padding='same',
            dropout_rate=0.00,
            filters_conv1d=3,
            activation_conv1d='linear',
            latent_sample_rate=42,
            pooler=torch.nn.AvgPool1d,
        )
    input = torch.randn(64,50,3)
    input = input.to(model.device)
    output = model(input)
    print(output.shape)