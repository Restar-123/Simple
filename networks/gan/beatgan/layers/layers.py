
import torch
from typing import List, Tuple, Type, Union


class AE(torch.nn.Module):
    """
    Simple AE Implementation
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, return_latent: bool = False):
        super(AE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.return_latent = return_latent

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.encoder(x)
        x_pred = self.decoder(z)

        if self.return_latent:
            return x_pred, z

        return x_pred

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

class ConvBlock(torch.nn.Module):
    def __init__(self, conv_layer, out_channels: int, activation, batch_norm: bool = False):
        super(ConvBlock, self).__init__()

        self.conv = conv_layer
        self.activation = activation
        self.norm = torch.nn.BatchNorm1d(out_channels) if batch_norm else torch.nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x, *args, **kwargs)))

class ConvEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, filters: List[int], conv_parameters: List[Tuple[int, int, int, bool, bool]],
                 block: Type[ConvBlock] = ConvBlock, conv_layer=torch.nn.Conv1d, activation=torch.nn.Identity(inplace = False)):
        super(ConvEncoder, self).__init__()

        dims = [input_dim] + filters
        modules = []
        for in_channels, out_channels, (kernel_size, stride, padding, bias, batch_norm) \
                in zip(dims[:-1], dims[1:], conv_parameters):
            layer = conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            modules.append(block(layer, out_channels, activation, batch_norm))

        self.layers = torch.nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int]]]:
        # x: (B, C, W)
        pre_conv_sizes = []
        for block in self.layers:
            pre_conv_sizes.append(x.shape[2:])
            x = block(x)

        return x, pre_conv_sizes


class ConvDecoder(torch.nn.Module):
    def __init__(self, input_dim: int, filters: List[int], conv_parameters: List[Tuple[int, int, int, bool, bool]],
                 block: Type[ConvBlock] = ConvBlock, conv_layer=torch.nn.ConvTranspose1d, activation=torch.nn.Identity(inplace=False)):
        super(ConvDecoder, self).__init__()

        dims = [input_dim] + filters
        modules = []
        for in_channels, out_channels, (kernel_size, stride, padding, bias, batch_norm) \
                in zip(dims[:-2], dims[1:-1], conv_parameters):
            layer = conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            modules.append(block(layer, out_channels, activation, batch_norm))

        # Don't use activation for last layer
        layer = conv_layer(dims[-2], dims[-1], conv_parameters[-1][0], conv_parameters[-1][1], conv_parameters[-1][2],
                           bias=conv_parameters[-1][3])
        modules.append(block(layer, dims[-1], torch.nn.Identity(), batch_norm=conv_parameters[-1][4]))

        self.layers = torch.nn.ModuleList(modules)

    def forward(self, inputs: Tuple[torch.Tensor, List[Tuple[int]]]) -> torch.Tensor:
        # x: (B, D_l, T_l)
        x, pre_conv_sizes = inputs
        for conv_block, pre_conv_size in zip(self.layers, pre_conv_sizes[::-1]):
            x = conv_block(x, output_size=pre_conv_size)

        return x


class BeatGANConvEncoder(ConvEncoder):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50,
                 last_kernel_size: int = 10, return_features: bool = False):
        filters = [
            conv_filters,
            conv_filters * 2,
            conv_filters * 4,
            conv_filters * 8,
            conv_filters * 16
        ]
        conv_params = [
            # (kernel_size, stride, padding, bias, batch_norm)
            (4, 2, 1, False, False),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
        ]
        super(BeatGANConvEncoder, self).__init__(input_dim, filters, conv_params, conv_layer=torch.nn.Conv1d,
                                                 activation=torch.nn.LeakyReLU(0.2, False))

        self.return_features = return_features

        self.last_conv = torch.nn.Conv1d(conv_filters * 16, latent_dim, last_kernel_size, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Union[List[Tuple[int]], torch.Tensor]]:
        x, sizes = super(BeatGANConvEncoder, self).forward(x)

        sizes.append(x.shape[2:])
        x_last = self.last_conv(x)

        if self.return_features:
            return x_last, x

        return x_last, sizes


class BeatGANConvDecoder(ConvDecoder):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50, last_kernel_size: int = 10):
        filters = [
            conv_filters * 16,
            conv_filters * 8,
            conv_filters * 4,
            conv_filters * 2,
            conv_filters,
            input_dim
        ]
        conv_params = [
            # (kernel_size, stride, padding, bias, batch_norm)
            (last_kernel_size, 1, 0, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
            (4, 2, 1, False, True),
        ]
        super(BeatGANConvDecoder, self).__init__(latent_dim, filters, conv_params, conv_layer=torch.nn.ConvTranspose1d,
                                                 activation=torch.nn.ReLU(inplace=False))

        # we replace tanh by sigmoid, because we do 0-1 normalization
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, inputs: Tuple[torch.Tensor, List[Tuple[int]]]) -> torch.Tensor:
        x = super(BeatGANConvDecoder, self).forward(inputs)
        return self.final_activation(x)


class BeatGANConvAE(AE):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50, last_kernel_size: int = 10):
        encoder = BeatGANConvEncoder(input_dim, conv_filters, latent_dim, last_kernel_size)
        decoder = BeatGANConvDecoder(input_dim, conv_filters, latent_dim, last_kernel_size)

        super(BeatGANConvAE, self).__init__(encoder, decoder, return_latent=False)