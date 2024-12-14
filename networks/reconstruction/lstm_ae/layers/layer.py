import abc
import functools
from typing import Tuple, Union, Callable, List, Type

import torch
import torch.nn.functional as F
import torch.nn as nn
from .myrnn import MyRNN

class AE(torch.nn.Module):
    """
    Simple AE Implementation
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, return_latent: bool = False):
        super(AE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.return_latent = return_latent

    def forward(self, x: torch.Tensor) :
        z = self.encoder(x)
        x_pred = self.decoder(z)

        if self.return_latent:
            return x_pred, z

class LSTMAEDecoder(torch.nn.Module, abc.ABC):
    def __init__(self, enc_hidden_dimension: int, hidden_dimensions: List[int], output_dimension: int):
        super().__init__()

    @abc.abstractmethod
    def forward(self, initial_hidden: torch.Tensor, seq_len: int, x: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class LSTMAEDecoderSimple(LSTMAEDecoder):
    """
    Reconstruct the time series using a LSTM decoder, starting with an initial hidden state from the encoder
    that is used as input to every timestep of the decoder
    This corresponds to Mirza 2018
    """
    def __init__(self, enc_hidden_dimension: int, hidden_dimensions: List[int], output_dimension: int):
        super().__init__(enc_hidden_dimension, hidden_dimensions, output_dimension)

        self.lstm = MyRNN(layer_type='LSTM', input_dimension=enc_hidden_dimension, model='s2s',
                        hidden_dimensions=hidden_dimensions + [output_dimension])

    def forward(self, initial_hidden: List[torch.Tensor], seq_len: int, x: torch.Tensor = None) -> torch.Tensor:
        """

        :param initial_hidden: list (length hidden_layers) of tensors of shape (B, D)
        :param seq_len: int that determines the length of the produced sequence
        :param x: The ground truth sequence that should be reconstructed as a tensor of shape (T, B, D).
         This will be fed into the LSTM during training instead of the output from the previous step.
        :return: Tensor of shape (T, B, D)
        """
        # Repeat the encoder output seq_len times
        hidden = torch.unsqueeze(initial_hidden[-1], dim=0)
        h_shape = list(hidden.shape)
        h_shape[0] = seq_len
        hidden = hidden.expand(*h_shape)

        result = self.lstm(hidden)
        return result


class LSTMAEDecoderReverse(LSTMAEDecoder):
    """
    Reconstruct the time series in the opposite direction, starting with an initial hidden state from the encoder
    This corresponds to Malhotra 2016
    """
    def __init__(self, enc_hidden_dimension: int, hidden_dimensions: List[int], output_dimension: int):
        super().__init__(enc_hidden_dimension, hidden_dimensions, output_dimension)

        self.lstm = MyRNN(layer_type='LSTM', input_dimension=output_dimension, model='s2as',
                        hidden_dimensions=[enc_hidden_dimension] + hidden_dimensions)
        self.linear = torch.nn.Linear(hidden_dimensions[-1], output_dimension)

    def forward(self, initial_hidden: List[torch.Tensor], seq_len: int, x: torch.Tensor = None) -> torch.Tensor:
        """

        :param initial_hidden: tensor of shape (B, D)
        :param seq_len: int that determines the length of the produced sequence
        :param x: The ground truth sequence that should be reconstructed as a tensor of shape (T, B, D).
         This will be fed into the LSTM during training instead of the output from the previous step.
        :return: Tensor of shape (T, B, D)
        """
        # Produce the last output
        hidden = initial_hidden[::-1]
        output = [self.linear(hidden[-1].unsqueeze(0))]
        hidden = (hidden, [torch.zeros_like(h) for h in hidden])

        if self.training and x is not None:
            # Use actual time series input instead of predictions
            # Use only x_2, ..., x_T as inputs, since \hat{x}_1 is predicted from x_2
            # Inputs are reversed, since we want to generate \hat{x}_T, \hat{x}_{T-1}, ..., \hat{x}_1
            inputs = torch.flip(x[1:], dims=(0,))
            result = self.lstm(inputs, hidden_states=hidden)[-1]
            # Apply linear layer
            result = self.linear(result)
            result = torch.cat(output + [result], dim=0)
        else:
            # Generate sequence from scratch
            inputs = output

            for t in range(1, seq_len):
                input = inputs[t-1]
                out, hidden = self.lstm(input, hidden_states=hidden, return_hidden=True)
                output.append(self.linear(out[-1]))
                hidden = [(h.squeeze(0), c.squeeze(0)) for h, c in hidden]

            result = torch.cat(output, dim=0)

        # Reverse output to get the correct order
        result = torch.flip(result, dims=(0,))

        return result
