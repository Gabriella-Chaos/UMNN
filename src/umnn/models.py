import torch
import torch.nn as nn
from .integral import UMNNNeuralIntegral

def flatten_params(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        # x: (N, 1) - Integration variable
        # h: (N, H_dim) - Context
        if h is not None:
            inp = torch.cat((x, h), 1)
        else:
            inp = x
        return self.net(inp) + 1.

class MonotonicNN(nn.Module):
    """
    Monotonic Neural Network using Numerical Integration.
    Transforms inputs x (Batch, Dim) monotonically w.r.t x, conditioned on h (Batch, Context).
    Uses a shared integrand for all dimensions of x, effectively treating them as a batch.
    """
    def __init__(self, context_dim, hidden_layers, nb_steps=50):
        super(MonotonicNN, self).__init__()
        # Integrand input: 1 (t) + context_dim
        self.integrand = IntegrandNN(1 + context_dim, hidden_layers)
        self.nb_steps = nb_steps
        
        # Context Network: Predicts scaling (alpha) and offset (beta) from h
        # alpha * Integral + beta
        hs = [context_dim] + hidden_layers + [2]
        net = []
        for h0, h1 in zip(hs[:-1], hs[1:]):
            net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        net.pop()
        self.context_net = nn.Sequential(*net)

    def forward(self, x, h):
        """
        x: (Batch, Dim)
        h: (Batch, ContextDim)
        """
        B, D = x.shape
        
        # 1. Compute Scaling and Offset from h
        # shape: (Batch, 2)
        out = self.context_net(h)
        offset = out[:, [0]].unsqueeze(2) # (Batch, 1, 1) -> Broadcast to D?
        scaling = torch.exp(out[:, [1]]).unsqueeze(2)
        
        # If we want specific scaling per dimension, context_net should output 2*D.
        # But assuming shared transformation logic:
        offset = out[:, 0].view(B, 1)
        scaling = torch.exp(out[:, 1].view(B, 1))

        # 2. Integrate
        # We handle the batch of dimensions by flattening.
        # x_flat: (Batch * Dim, 1)
        x_flat = x.reshape(-1, 1)
        x0_flat = torch.zeros_like(x_flat)
        
        # h needs to be repeated for each dimension
        # h: (Batch, C) -> (Batch, Dim, C) -> (Batch*Dim, C)
        if h is not None:
            C = h.shape[1]
            h_flat = h.unsqueeze(1).expand(B, D, C).reshape(-1, C)
        else:
            h_flat = None
            
        flat_params = flatten_params(self.integrand.parameters())
        
        # Run Integral
        # Result: (Batch * Dim, 1)
        int_val = UMNNNeuralIntegral.apply(x0_flat, x_flat, self.integrand, flat_params, h_flat, self.nb_steps)
        
        # Reshape back
        int_val = int_val.view(B, D)
        
        return scaling * int_val + offset