import torch
import torch.nn as nn
import unittest
import pytest
import numpy as np

from umnn.models import MonotonicNN, ParallelMonotonicNN, GeneralizedUMNN
from umnn.integral import UMNNNeuralIntegral
from umnn.triton_ops import is_triton_available

def flatten_params(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class QuadraticIntegrand(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, h=None):
        return x ** 2

class TestUMNNComprehensive(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

    def test_integral_numerical_accuracy(self):
        integrand = QuadraticIntegrand().to(self.device)
        nb_steps = 20
        
        x = torch.tensor([[1.0], [2.0], [3.0]], device=self.device)
        x0 = torch.zeros_like(x)
        h = None
        flat_params = flatten_params(integrand.parameters())
        expected = (x ** 3) / 3.0
        
        result = UMNNNeuralIntegral.apply(x0, x, integrand, flat_params, h, nb_steps)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_gradcheck(self):
        integrand = QuadraticIntegrand().to(self.device).double()
        nb_steps = 10
        # Correctly initialize leaf tensors on device
        x = torch.tensor([[1.5], [0.5]], dtype=torch.double, device=self.device, requires_grad=True)
        x0 = torch.tensor([[0.0], [0.0]], dtype=torch.double, device=self.device, requires_grad=True)
        h = None
        flat_params = flatten_params(integrand.parameters())
        flat_params.requires_grad_(True)
        
        def func(x0_, x_, params_):
            return UMNNNeuralIntegral.apply(x0_, x_, integrand, params_, h, nb_steps)
        
        # gradcheck will use float64, which now bypasses Triton to avoid compilation errors
        test = torch.autograd.gradcheck(func, (x0, x, flat_params), eps=1e-6, atol=1e-4)
        self.assertTrue(test)

    def test_context_dependence(self):
        model = MonotonicNN(context_dim=2, hidden_layers=[10], nb_steps=10).to(self.device)
        x = torch.randn(5, 1, device=self.device)
        h1 = torch.zeros(5, 2, device=self.device)
        h2 = torch.ones(5, 2, device=self.device)
        
        y1 = model(x, h1)
        y2 = model(x, h2)
        diff = (y1 - y2).abs().sum()
        self.assertGreater(diff.item(), 1e-5)

    def test_multidimensional_input(self):
        D = 4
        model = MonotonicNN(context_dim=2, hidden_layers=[10], nb_steps=5).to(self.device)
        # Correct initialization for leaf tensors
        x = torch.randn(10, D, device=self.device, requires_grad=True)
        h = torch.randn(10, 2, device=self.device, requires_grad=True)
        
        y = model(x, h)
        self.assertEqual(y.shape, (10, D))
        y.sum().backward()
        
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertIsNotNone(h.grad)

    def test_parallel_monotonic_nn(self):
        """Test ParallelMonotonicNN with independent weights."""
        D = 3
        model = ParallelMonotonicNN(num_dims=D, context_dim=2, hidden_layers=[10]).to(self.device)
        x = torch.randn(5, D, device=self.device)
        h = torch.randn(5, 2, device=self.device)
        
        y = model(x, h)
        self.assertEqual(y.shape, (5, D))
        
        # Check independence: modifying x[:, 0] should not affect y[:, 1]
        x_mod = x.clone()
        x_mod[:, 0] += 1.0
        y_mod = model(x_mod, h)
        
        diff_0 = (y_mod[:, 0] - y[:, 0]).abs().sum()
        diff_1 = (y_mod[:, 1] - y[:, 1]).abs().sum()
        
        self.assertGreater(diff_0.item(), 1e-5)
        self.assertLess(diff_1.item(), 1e-5, "Output dimension 1 affected by input dimension 0!")

    def test_generalized_umnn(self):
        """Test GeneralizedUMNN (D -> 1)."""
        D = 4
        model = GeneralizedUMNN(num_dims=D, context_dim=2, hidden_layers=[10]).to(self.device)
        x = torch.randn(5, D, device=self.device, requires_grad=True)
        h = torch.randn(5, 2, device=self.device, requires_grad=True)
        
        y = model(x, h)
        self.assertEqual(y.shape, (5, 1))
        
        # Test backward
        y.sum().backward()
        
        # Check params have grad
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.sum() != 0:
                has_grad = True
                break
        self.assertTrue(has_grad)

    @pytest.mark.skipif(not is_triton_available(), reason="Triton not available")
    def test_triton_vs_torch_consistency(self):
        """
        Check consistency between Triton and PyTorch implementations.
        """
        if self.device == "cpu":
            return

        integrand = QuadraticIntegrand().to(self.device)
        nb_steps = 20
        x = torch.tensor([[1.0], [2.0]], device=self.device)
        x0 = torch.zeros_like(x)
        h = None
        flat_params = flatten_params(integrand.parameters())
        
        # By default (float32, cuda, triton avail) -> Triton
        res_triton = UMNNNeuralIntegral.apply(x0, x, integrand, flat_params, h, nb_steps)
        
        # Force fallback by using double
        x_d = x.double()
        x0_d = x0.double()
        integrand_d = QuadraticIntegrand().to(self.device).double()
        flat_params_d = flatten_params(integrand_d.parameters())
        
        res_torch_d = UMNNNeuralIntegral.apply(x0_d, x_d, integrand_d, flat_params_d, h, nb_steps)
        
        self.assertTrue(torch.allclose(res_triton.double(), res_torch_d, atol=1e-5))

if __name__ == '__main__':
    unittest.main()