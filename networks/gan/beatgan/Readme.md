# 核心是ConvEnCoder

```angular2html
class BeatGANModel(GAN, nn.Module):
    def __init__(self, input_dim: int, device=0, conv_filters: int = 32, latent_dim: int = 50, last_kernel_size: int = 10):
        # Note: BeatGAN will only work with a window size of exactly 320
        generator = BeatGANConvAE(input_dim, conv_filters, latent_dim, last_kernel_size)
        discriminator = BeatGANConvEncoder(input_dim, conv_filters, 1, last_kernel_size, return_features=True)
```

```angular2html
class BeatGANConvAE(AE):
    def __init__(self, input_dim: int, conv_filters: int = 32, latent_dim: int = 50, last_kernel_size: int = 10):
        encoder = BeatGANConvEncoder(input_dim, conv_filters, latent_dim, last_kernel_size)
        decoder = BeatGANConvDecoder(input_dim, conv_filters, latent_dim, last_kernel_size)
```
