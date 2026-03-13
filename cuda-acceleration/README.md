# CUDA Acceleration

Custom CUDA/Triton kernels for deep learning: GEMM, flash attention, softmax, layer norm.

## Features
- Tiled SGEMM with shared memory and double buffering
- Triton fused softmax kernel
- Triton FlashAttention forward/backward
- Warp-level parallel reduction
- INT8 quantized GEMM kernel

## Usage
```python
from src.triton_kernels import fused_softmax, flash_attention_triton
out = fused_softmax(x)  # faster than torch.softmax for large tensors
```
