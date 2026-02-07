# UMNN - Unified Monotonic Neural Networks

A fast and unified implementation of UMNN using PyTorch and Triton.

## Features

- **Monotonic Neural Networks**: Invertible transformations based on numerical integration.
- **Triton Optimization**: Custom Triton kernels for efficient Clenshaw-Curtis quadrature and gradient computation.
- **Batching Support**: Efficiently handles multidimensional inputs.
- **Generalized UMNN**: Supports deep multidimensional architectures with independent transformations per dimension.
- **Modern Architecture**: Clean `src` layout with `pyproject.toml` configuration.

## Hardware Requirements

- **GPU Acceleration**: Requires an NVIDIA GPU (CUDA) or AMD GPU (ROCm) to run Triton kernels.
- **CPU Fallback**: Automatically falls back to optimized PyTorch operations on machines without a GPU or on Windows.
- **OS**: Triton kernels are primarily supported on **Linux**.

## Installation

### Install from Source

```bash
git clone https://github.com/Gabriella-Chaos/UMNN.git
cd UMNN
pip install .
```

*Note: Triton will be installed automatically if you are on Linux.*

To install for development (editable mode):

```bash
pip install -e .[dev]
```

## Usage

### Simple 1D Monotonic Transformation (Shared Weights)
```python
import torch
from umnn import MonotonicNN

model = MonotonicNN(context_dim=10, hidden_layers=[50, 50], nb_steps=20)
x = torch.randn(32, 4) # Batch=32, Dim=4
h = torch.randn(32, 10) # Context
y = model(x, h)
```

### Generalized Multidimensional UMNN (Independent Weights)
```python
from umnn import GeneralizedUMNN

# Maps D -> 1 using independent transformations per dimension
model = GeneralizedUMNN(num_dims=4, context_dim=10, hidden_layers=[50, 50])
y = model(x, h) # Output shape: (32, 1)
```

## Testing

Run tests using `pytest`:

```bash
pytest
```

The test suite automatically detects hardware and verifies both Triton (if available) and PyTorch fallback paths.

## Structure

- `src/umnn/`: Source code package.
- `src/umnn/triton_ops.py`: Custom Triton kernels for reduction and expansion.
- `tests/`: Comprehensive numerical and autograd tests.
- `pyproject.toml`: Build configuration and dependencies.
