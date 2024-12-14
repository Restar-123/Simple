import torch
from typing import List, Tuple, Type, Union

class GAN(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        z, real_x = inputs
        # Generate x from z
        fake_x = self.generator(z)

        real_x_score = self.discriminator(real_x)
        fake_x_score = self.discriminator(fake_x)

        return fake_x, real_x_score, fake_x_score


class TADGANEncoder(torch.nn.Module):
    def __init__(self, input_size: int, window_size: int, lstm_hidden_size: int = 100, latent_size: int = 20):
        super(TADGANEncoder, self).__init__()

        self.lstm = torch.nn.LSTM(input_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(2 * lstm_hidden_size * window_size, latent_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, T, D)

        x, _ = self.lstm(x)
        # (B, T, 2*hidden_lstm)
        x = x.reshape(x.shape[0], -1)
        # (B, 2*T*hidden_lstm)
        x = self.linear(x)
        # (B, latent)

        return x


class TADGANGenerator(torch.nn.Module):
    def __init__(self, window_size: int, output_size: int, latent_size: int = 20, lstm_hidden_size: int = 64,
                 dropout: float = 0.2):
        super(TADGANGenerator, self).__init__()

        self.linear1 = torch.nn.Linear(latent_size, window_size // 2)
        self.lstm1 = torch.nn.LSTM(1, lstm_hidden_size, bidirectional=True, batch_first=True, dropout=dropout)
        self.upsample = torch.nn.Upsample(size=window_size, mode='nearest')
        self.lstm2 = torch.nn.LSTM(2 * lstm_hidden_size, lstm_hidden_size, bidirectional=True, batch_first=True,
                                   dropout=dropout)
        self.linear2 = torch.nn.Linear(2 * lstm_hidden_size, output_size)
        # We use Sigmoid as the final activation function, because we normalize data to [0, 1] instead of [-1, 1]
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, latent)

        x = self.linear1(x).unsqueeze(-1)
        # (B, T // 2, 1)
        x, _ = self.lstm1(x)
        # (B, T // 2, 2*hidden_lstm)
        x = self.upsample(x.transpose(1, 2)).transpose(1, 2)
        # (B, T, 2*hidden_lstm)
        x, _ = self.lstm2(x)
        # (B, T, 2*hidden_lstm)
        # Apply linear layer at each time step independently
        x = self.linear2(x)
        # (B, T, out)
        x = self.final_activation(x)

        return x


class TADGANDiscriminatorX(torch.nn.Module):
    def __init__(self, input_size: int, window_size: int, conv_filters: int = 64, conv_kernel_size: int = 5,
                 dropout: float = 0.25):
        super(TADGANDiscriminatorX, self).__init__()

        self.conv1 = torch.nn.Conv1d(input_size, conv_filters, conv_kernel_size)
        self.conv2 = torch.nn.Conv1d(conv_filters, conv_filters, conv_kernel_size)
        self.conv3 = torch.nn.Conv1d(conv_filters, conv_filters, conv_kernel_size)
        self.conv4 = torch.nn.Conv1d(conv_filters, conv_filters, conv_kernel_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        out_length = window_size - 4 * (conv_kernel_size - 1)
        self.classification = torch.nn.Linear(out_length * conv_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, T, D)

        x = x.transpose(1, 2)  # Needed for conv layers
        # (B, D, T)
        x = self.dropout(self.leakyrelu(self.conv1(x)))
        # (B, conv_filters, T - (conv_kernel - 1))
        x = self.dropout(self.leakyrelu(self.conv2(x)))
        # (B, conv_filters, T - 2*(conv_kernel - 1))
        x = self.dropout(self.leakyrelu(self.conv3(x)))
        # (B, conv_filters, T - 3*(conv_kernel - 1))
        x = self.dropout(self.leakyrelu(self.conv4(x)))
        # (B, conv_filters, T - 4*(conv_kernel - 1))

        x = self.classification(x.view(x.shape[0], -1))
        # (B, 1)

        return x


class TADGANDiscriminatorZ(torch.nn.Module):
    def __init__(self, latent_size: int, hidden_size: int = 20, dropout: float = 0.2):
        super(TADGANDiscriminatorZ, self).__init__()

        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        self.classification = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (B, latent)

        x = self.dropout(self.leakyrelu(self.linear1(x)))
        # (B, hidden)
        x = self.dropout(self.leakyrelu(self.linear2(x)))
        # (B, hidden)

        x = self.classification(x.view(x.shape[0], -1))
        # (B, 1)

        return x

