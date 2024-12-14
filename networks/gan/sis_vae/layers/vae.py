import abc
from typing import Tuple, Sequence


from typing import Union, Callable, Sequence

import torch


class MLP(torch.nn.Module):
    def __init__(self, input_features: int, hidden_layers: Union[int, Sequence[int]], output_features: int,
                 activation: Callable = torch.nn.Identity(), activation_after_last_layer: bool = False):
        super(MLP, self).__init__()

        self.activation = activation
        self.activation_after_last_layer = activation_after_last_layer

        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]

        layers = [input_features] + list(hidden_layers) + [output_features]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(inp, out) for inp, out in zip(layers[:-1], layers[1:])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)
            out = self.activation(out)

        out = self.layers[-1](out)
        if self.activation_after_last_layer:
            out = self.activation(out)

        return out
def sample_normal(mu: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False, num_samples: int = 1):
    # ln(σ) = 0.5 * ln(σ^2) -> σ = e^(0.5 * ln(σ^2))
    if log_var:
        sigma = std_or_log_var.mul(0.5).exp_()
    else:
        sigma = std_or_log_var

    if num_samples == 1:
        eps = torch.randn_like(mu)  # also copies device from mu
    else:
        eps = torch.rand((num_samples,) + mu.shape, dtype=mu.dtype, device=mu.device)
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
    # z = μ + σ * ϵ, with ϵ ~ N(0,I)
    return eps.mul(sigma).add_(mu)


def normal_standard_normal_kl(mean: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        kl_loss = torch.sum(1 + std_or_log_var - mean.pow(2) - std_or_log_var.exp(), dim=-1)
    else:
        kl_loss = torch.sum(1 + torch.log(std_or_log_var.pow(2)) - mean.pow(2) - std_or_log_var.pow(2), dim=-1)
    return -0.5 * kl_loss


def normal_normal_kl(mean_1: torch.Tensor, std_or_log_var_1: torch.Tensor, mean_2: torch.Tensor,
                     std_or_log_var_2: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        return 0.5 * torch.sum(std_or_log_var_2 - std_or_log_var_1 + (torch.exp(std_or_log_var_1)
                               + (mean_1 - mean_2)**2) / torch.exp(std_or_log_var_2) - 1, dim=-1)

    return torch.sum(torch.log(std_or_log_var_2) - torch.log(std_or_log_var_1) \
                     + 0.5 * (std_or_log_var_1**2 + (mean_1 - mean_2)**2) / std_or_log_var_2**2 - 0.5, dim=-1)


class DenseVAEEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (100, 100), latent_dim: int = 10,
                 activation=torch.nn.ReLU()):
        super(DenseVAEEncoder, self).__init__()

        self.latent_dim = latent_dim

        self.mlp = MLP(input_dim, hidden_dims, 2*latent_dim, activation=activation, activation_after_last_layer=False)
        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        mlp_out = self.mlp(x)

        mean, std = mlp_out.tensor_split(2, dim=-1)
        std = self.softplus(std)

        return mean, std


class VAE(torch.nn.Module):
    """
    VAE Implementation that supports normal distribution with diagonal cov matrix in the latent space
    and the output
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, logvar_out: bool = True):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.log_var = logvar_out

    def forward(self, x: torch.Tensor, return_latent_sample: bool = False, num_samples: int = 1,
                force_sample: bool = False) -> Tuple[torch.Tensor, ...]:
        z_mu, z_std_or_log_var = self.encoder(x)

        if self.training or num_samples > 1 or force_sample:
            z_sample = sample_normal(z_mu, z_std_or_log_var, log_var=self.log_var, num_samples=num_samples)
        else:
            z_sample = z_mu

        x_dec_mean, x_dec_std = self.decoder(z_sample)

        if not return_latent_sample:
            return z_mu, z_std_or_log_var, x_dec_mean, x_dec_std

        return z_mu, z_std_or_log_var, x_dec_mean, x_dec_std, z_sample

class DistVAE(torch.nn.Module):
    """
    VAE Implementation that supports arbitrary torch.distributions as long as the latent distribution supports
    rsample
    """

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super(DistVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def get_latent_prior(self, shape: torch.Size):
        return torch.distributions.Normal(0, 1).expand(shape)

    def sample_prior(self, shape: torch.Size):
        z_sample = self.get_latent_prior(shape).sample()
        x_dist = self.decoder(z_sample)
        x_sample = x_dist.sample()

        return z_sample, x_dist, x_sample

    def sample_q(self, x: torch.Tensor):
        latent_dist = self.encoder(x)
        z_sample = latent_dist.sample()
        x_dist = self.decoder(z_sample)
        x_sample = x_dist.sample()

        return latent_dist, z_sample, x_dist, x_sample

    def forward(self, x: torch.Tensor, return_latent_sample: bool = False, num_samples: int = 1,
                force_sample: bool = False) -> Tuple[torch.distributions.Distribution, ...]:
        latent_dist = self.encoder(x)

        if self.training or num_samples > 1 or force_sample:
            if not latent_dist.has_rsample:
                raise RuntimeError('Latent distribution must support rsample to propagate gradients!')

            sample_shape = (num_samples,) if num_samples > 1 else None
            z_sample = latent_dist.rsample(sample_shape)
        else:
            z_sample = latent_dist.mean

        data_dist = self.decoder(z_sample)

        if not return_latent_sample:
            return latent_dist, data_dist, self.get_latent_prior(z_sample.shape)

        return latent_dist, data_dist, self.get_latent_prior(z_sample.shape), z_sample


class NormalVAEEncoder(abc.ABC, torch.nn.Module):
    """
    Generic class that returns a normal distribution parameterized by mean and standard deviation.
    Subclasses should override `internal_forward` to generate these parameters from some input x
    """
    @abc.abstractmethod
    def internal_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        mean, std = self.internal_forward(x)
        dist = torch.distributions.Normal(mean, std)

        return dist


class DenseNormalVAEEncoder(NormalVAEEncoder):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (100, 100), latent_dim: int = 10,
                 activation=torch.nn.ReLU()):
        super(DenseNormalVAEEncoder, self).__init__()

        self.latent_dim = latent_dim

        self.mlp = MLP(input_dim, hidden_dims, 2*latent_dim, activation=activation, activation_after_last_layer=False)
        self.softplus = torch.nn.Softplus()

    def internal_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        mlp_out = self.mlp(x)

        mean, std = mlp_out.tensor_split(2, dim=-1)
        std = self.softplus(std)

        return mean, std

