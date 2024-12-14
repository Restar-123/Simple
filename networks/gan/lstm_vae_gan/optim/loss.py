"""
Abstract class implementing the general interface of a loss.
"""
import abc
import math
from typing import Tuple, Union

import torch
from torch.nn import functional as F


class Loss(torch.nn.modules.loss._Loss, abc.ABC):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(Loss, self).__init__(size_average, reduce, reduction)

    @abc.abstractmethod
    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        raise NotImplementedError()


def normal_normal_kl(mean_1: torch.Tensor, std_or_log_var_1: torch.Tensor, mean_2: torch.Tensor,
                     std_or_log_var_2: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        return 0.5 * torch.sum(std_or_log_var_2 - std_or_log_var_1 + (torch.exp(std_or_log_var_1)
                               + (mean_1 - mean_2)**2) / torch.exp(std_or_log_var_2) - 1, dim=-1)

    return torch.sum(torch.log(std_or_log_var_2) - torch.log(std_or_log_var_1) \
                     + 0.5 * (std_or_log_var_1**2 + (mean_1 - mean_2)**2) / std_or_log_var_2**2 - 0.5, dim=-1)

def normal_standard_normal_kl(mean: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        kl_loss = torch.sum(1 + std_or_log_var - mean.pow(2) - std_or_log_var.exp(), dim=-1)
    else:
        kl_loss = torch.sum(1 + torch.log(std_or_log_var.pow(2)) - mean.pow(2) - std_or_log_var.pow(2), dim=-1)
    return -0.5 * kl_loss


class VAELoss(Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', logvar_out: bool = True):
        super(VAELoss, self).__init__(size_average, reduce, reduction)
        self.logvar_out = logvar_out

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        z_mean, z_std_or_log_var, x_dec_mean, x_dec_std = predictions[:4]
        if len(predictions) > 4:
            z_prior_mean, z_prior_std_or_logvar = predictions[4:]
        else:
            z_prior_mean, z_prior_std_or_logvar = None, None

        y, = targets

        # Gaussian nnl loss assumes multivariate normal with diagonal sigma
        # Alternatively we can use torch.distribution.Normal(x_dec_mean, x_dec_std).log_prob(y).sum(-1)
        # or torch.distribution.MultivariateNormal(mean, cov).log_prob(y).sum(-1)
        # with cov = torch.eye(feat_dim).repeat([1,bz,1,1])*std.pow(2).unsqueeze(-1).
        # However setting up a distribution seems to be an unnecessary computational overhead.
        # However, this requires pytorch version > 1.9!!!
        nll_gauss = F.gaussian_nll_loss(x_dec_mean, y, x_dec_std.pow(2), reduction='none').sum(-1)
        # For pytorch version < 1.9 use:
        # nll_gauss = -torch.distribution.Normal(x_dec_mean, x_dec_std).log_prob(y).sum(-1)

        # get KL loss
        if z_prior_mean is None and z_prior_std_or_logvar is None:
            # If a prior is not given, we assume standard normal
            kl_loss = normal_standard_normal_kl(z_mean, z_std_or_log_var, log_var=self.logvar_out)
        else:
            if z_prior_mean is None:
                z_prior_mean = torch.tensor(0, dtype=z_mean.dtype, device=z_mean.device)
            if z_prior_std_or_logvar is None:
                value = 0 if self.logvar_out else 1
                z_prior_std_or_logvar = torch.tensor(value, dtype=z_std_or_log_var.dtype, device=z_std_or_log_var.device)

            kl_loss = normal_normal_kl(z_mean, z_std_or_log_var, z_prior_mean, z_prior_std_or_logvar,
                                       log_var=self.logvar_out)

        # Combine
        final_loss = nll_gauss + kl_loss

        if self.reduction == 'none':
            return final_loss
        elif self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)

class GANDiscriminatorLoss(Loss):
    def __init__(self):
        """
        This is the original GAN loss, i.e., - E[log(D(x))] - E[log(1 - D(G(z)))]
        """
        super(GANDiscriminatorLoss, self).__init__()

        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, fake_x_score = predictions

        loss_real = self.cross_entropy(real_x_score, torch.ones_like(real_x_score))
        loss_fake = self.cross_entropy(fake_x_score, torch.zeros_like(fake_x_score))

        return loss_real + loss_fake


class GANGeneratorLoss(Loss):
    def __init__(self):
        """
        This is the original GAN loss, i.e., E[log(1 - D(G(z)))]
        """
        super(GANGeneratorLoss, self).__init__()

        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions: Tuple[torch.Tensor, ...],
                targets: Tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor:
        fake_x, real_x_score, fake_x_score = predictions

        loss_fake = self.cross_entropy(fake_x_score, torch.zeros_like(fake_x_score))

        return -loss_fake

class MaskedVAELoss(VAELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MaskedVAELoss, self).__init__(size_average, reduce, reduction, logvar_out=False)

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        mean_z, std_z, mean_x, std_x, sample_z, mask = predictions
        actual_x, = targets

        if mask is None:
            mean_z = mean_z.unsqueeze(1)
            std_z = std_z.unsqueeze(1)
            return super(MaskedVAELoss, self).forward((mean_z, std_z, mean_x, std_x), (actual_x,), *args, **kwargs)

        # If the loss is masked, one of the terms in the kl loss is weighted, so we can't compute it exactly
        # anymore and have to use a MC approximation like for the output likelihood
        nll_output = torch.sum(mask * F.gaussian_nll_loss(mean_x, actual_x, std_x**2, reduction='none'), dim=-1)

        # This is p(z), i.e., the prior likelihood of Z. The paper assumes p(z) = N(z| 0, I), we drop constants
        beta = torch.mean(mask, dim=(1, 2)).unsqueeze(-1)
        nll_prior = beta * 0.5 * torch.sum(sample_z * sample_z, dim=-1, keepdim=True)

        nll_approx = torch.sum(F.gaussian_nll_loss(mean_z, sample_z, std_z**2, reduction='none'), dim=-1, keepdim=True)

        final_loss = nll_output + nll_prior - nll_approx

        if self.reduction == 'none':
            return final_loss
        elif self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)