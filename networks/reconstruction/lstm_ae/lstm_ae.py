import functools
from typing import Tuple, Union, Callable, List, Type
from .layers.layer import AE,LSTMAEDecoder,LSTMAEDecoderReverse,LSTMAEDecoderSimple
import torch
import torch.nn as nn

from .layers.myrnn import MyRNN
from common.utils import set_device
import logging
import time
import numpy as np


def getitem(x, item):
    return x[item]

def max_pool(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.max(x, dim=dim)[0]


class LSTM_AE(AE, nn.Module):
    """
    Generic LSTMAE implementation
    """
    def __init__(self, input_dimension: int, device = 0, hidden_dimensions=None, latent_pooling: Union[str, Callable] = 'last',
                 decoder_class: Type[LSTMAEDecoder] = LSTMAEDecoderReverse, return_latent: bool = False):
        if hidden_dimensions is None:
            hidden_dimensions = [40]

        encoder = MyRNN(layer_type='LSTM', input_dimension=input_dimension, model='s2as',
                        hidden_dimensions=hidden_dimensions)
        dec_hidden_dimensions = hidden_dimensions if len(hidden_dimensions) == 1 else hidden_dimensions[-2::-1]
        decoder = decoder_class(enc_hidden_dimension=hidden_dimensions[-1], hidden_dimensions=dec_hidden_dimensions,
                                output_dimension=input_dimension)

        super().__init__(encoder, decoder, return_latent=return_latent)

        if latent_pooling == 'last':
            self.latent_pooling = functools.partial(getitem, item=-1)
        elif latent_pooling == 'mean':
            self.latent_pooling = functools.partial(torch.mean, dim=0)
        elif latent_pooling == 'max':
            self.latent_pooling = functools.partial(max_pool, dim=0)
        elif callable(latent_pooling):
            self.latent_pooling = latent_pooling
        else:
            raise ValueError(f'Pooling method "{latent_pooling}" is not supported!')

        self.device = set_device(device)
        self.to(self.device)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        hidden = self.encoder(x)

        return [self.latent_pooling(h) for h in hidden]

    def forward(self, x) :
        """

        :param inputs: Tuple with a single tensor of shape (T, B, D)
        :return: tensor of shape (T, B, D)
        """
        seq_len = x.shape[0]
        hidden = self.encode(x)

        if self.training:
            pred = self.decoder(hidden, seq_len, x)
        else:
            pred = self.decoder(hidden, seq_len)

        if self.return_latent:
            return pred, hidden

        return pred

    def fit(self, train_loader,val_loader, epochs, lr, criterion=nn.MSELoss()):
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


class LSTMAEMalhotra2016(LSTM_AE):
    """
    Implementation of Malhotra 2016 (https://arxiv.org/pdf/1607.00148.pdf, default parameters)
    """
    def __init__(self, input_dimension: int, hidden_dimensions=None):
        super(LSTMAEMalhotra2016, self).__init__(input_dimension, hidden_dimensions=hidden_dimensions,
                                                 latent_pooling='last', decoder_class=LSTMAEDecoderReverse,
                                                 return_latent=False)


class LSTMAEMirza2018(LSTM_AE):
    """
    Mirza 2018 (http://repository.bilkent.edu.tr/bitstream/handle/11693/50234/Computer_network_intrusion_detection_using_sequential_LSTM_neural_networks_autoencoders.pdf?sequence=1)
    """
    def __init__(self, input_dimension: int, hidden_dimensions: List[int] = [64], latent_pooling: str = 'mean'):
        super(LSTMAEMirza2018, self).__init__(input_dimension, hidden_dimensions=hidden_dimensions,
                                              latent_pooling=latent_pooling, decoder_class=LSTMAEDecoderSimple,
                                              return_latent=False)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        """

        :param inputs: Tuple with a single tensor of shape (T, B, D)
        :return: tensor of shape (T, B, D)
        """
        pred = super(LSTMAEMirza2018, self).forward(input)
        pred = self.sigmoid(pred)

        return pred