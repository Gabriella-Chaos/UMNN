import torch
import unittest
import pytest

from umnn.models import MonotonicNN
from umnn.integral import UMNNNeuralIntegral
from umnn.triton_ops import is_triton_available

class TestUMNN(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def test_forward_shape(self):
        """Test that the forward pass returns the correct shape."""
        model = MonotonicNN(context_dim=5, hidden_layers=[10, 10], nb_steps=10).to(self.device)
        x = torch.randn(3, 4).to(self.device)
        h = torch.randn(3, 5).to(self.device)
        
        y = model(x, h)
        self.assertEqual(y.shape, (3, 4))
        
    def test_monotonicity(self):
        """Test that the output is monotonic with respect to input x."""
        model = MonotonicNN(context_dim=2, hidden_layers=[10], nb_steps=20).to(self.device)
        # Use Batch=10, Dim=1 to test monotonicity of 1D transformation
        x = torch.linspace(-1, 1, 10).view(10, 1).to(self.device)
        h = torch.randn(10, 2).to(self.device)
        
        # To strictly test monotonicity, fix h across batch
        h = h[0:1].expand(10, 2)
        
        y = model(x, h)
        
        y_np = y.detach().cpu().numpy().flatten()
        # Check if non-decreasing
        is_sorted = (y_np[1:] >= y_np[:-1]).all()
        self.assertTrue(is_sorted, f"Output not monotonic: {y_np}")
        
    def test_backward(self):
        """Test that gradients are computed for all inputs and parameters."""
        model = MonotonicNN(context_dim=2, hidden_layers=[10], nb_steps=5).to(self.device)
        x = torch.randn(2, 2, requires_grad=True).to(self.device)
        h = torch.randn(2, 2, requires_grad=True).to(self.device)
        
        y = model(x, h)
        loss = y.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad, "Input x gradient is None")
        self.assertIsNotNone(h.grad, "Context h gradient is None")
        
        # Check params grad
        for i, p in enumerate(model.parameters()):
            self.assertIsNotNone(p.grad, f"Parameter {i} gradient is None")

if __name__ == '__main__':
    unittest.main()