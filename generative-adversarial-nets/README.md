# Generative Adversarial Networks

GANs research: StyleGAN3, Progressive Growing, CycleGAN, Pix2Pix, conditional image synthesis.

## Features
- StyleGAN3 alias-free generator with modulated convolutions
- Progressive GAN with equalized learning rate
- CycleGAN unsupervised image-to-image translation
- Pix2Pix paired conditional synthesis with PatchGAN discriminator
- Spectral normalization and gradient penalty (R1/WGAN-GP)

## Usage
```python
from src.stylegan3 import StyleGAN3Generator, StyleGAN3Discriminator
G = StyleGAN3Generator(z_dim=512, w_dim=512, img_resolution=256)
img = G(z=torch.randn(1, 512), c=None)
```
