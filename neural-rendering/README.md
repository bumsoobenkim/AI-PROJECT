# Neural Rendering

Novel view synthesis: NeRF, Instant-NGP, 3D Gaussian Splatting.

## Features
- Vanilla NeRF with positional encoding and volume rendering
- Instant-NGP hash encoding for fast training
- 3D Gaussian Splatting differentiable rasterizer
- Hierarchical sampling with coarse/fine networks

## Usage
```python
from src.nerf import NeRF, VolumeRenderer
nerf = NeRF(pos_encoding_levels=10)
renderer = VolumeRenderer(nerf, near=2.0, far=6.0)
```
