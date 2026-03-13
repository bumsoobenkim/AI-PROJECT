# Diffusion-Based Graphics

Text-to-image and image editing: Stable Diffusion, ControlNet, IP-Adapter, latent diffusion.

## Features
- Latent Diffusion Model with VQVAE encoder/decoder
- DDPM/DDIM noise schedulers
- ControlNet conditioning (edges, depth, pose)
- IP-Adapter image prompt injection

## Usage
```python
from src.ldm import LatentDiffusionModel
ldm = LatentDiffusionModel(vae, unet, text_encoder)
img = ldm.generate("a sunset over mountains", steps=50)
```
