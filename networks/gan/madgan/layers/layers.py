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