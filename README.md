# UMNN - Unified Monotonic Neural Networks

A fast and unified implementation of UMNN using PyTorch and Triton.

## Features

- **Monotonic Neural Networks**: Invertible transformations based on numerical integration.
- **Triton Optimization**: Custom Triton kernels for efficient Clenshaw-Curtis quadrature and gradient computation (requires GPU).
- **Batching Support**: Efficiently handles multidimensional inputs.
- **Modern Architecture**: Clean `src` layout with `pyproject.toml` configuration.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Triton (for GPU acceleration, optional but recommended)
  - Triton is usually available on Linux.
  - On Windows, the package falls back to pure PyTorch.

### Install from Source

```bash
git clone https://github.com/your-repo/umnn.git
cd umnn
pip install .
```

To install for development (editable mode):

```bash
pip install -e .[dev]
```

## Usage

```python
import torch
from umnn import MonotonicNN

# Define model
# context_dim: Dimension of conditioning vector h
# hidden_layers: List of hidden layer sizes for the integrand
# nb_steps: Number of integration steps
model = MonotonicNN(context_dim=10, hidden_layers=[50, 50], nb_steps=20)

# Move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Inputs
x = torch.randn(32, 4).to(device) # Batch=32, Dim=4
h = torch.randn(32, 10).to(device) # Batch=32, Context=10

# Forward
y = model(x, h)

# Backward
loss = y.sum()
loss.backward()
```

## Testing

Run tests using `pytest`:

```bash
pytest
```

To test Triton kernels specifically (requires Linux + GPU):
The tests automatically detect if Triton is available.

## Structure

- `src/umnn/`: Source code package.
- `tests/`: Unit tests.
- `pyproject.toml`: Build configuration and dependencies.