import torch
import torch.nn as nn
from .integral import UMNNNeuralIntegral

def flatten_params(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class GroupedLinear(nn.Module):
    """
    Applies independent linear transformations to groups of inputs.
    Equivalent to D independent Linear layers running in parallel.
    """
    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, groups=groups)
        
    def forward(self, x):
        # x: (Batch, Channels) -> (Batch, Channels, 1)
        x = x.unsqueeze(2)
        out = self.conv(x)
        return out.squeeze(2)

class IntegrandNN(nn.Module):
    """
    Simple MLP Integrand for shared weights (MonotonicNN).
    """
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        if h is not None:
            inp = torch.cat((x, h), 1)
        else:
            inp = x
        return self.net(inp) + 1.

class ParallelIntegrandNN(nn.Module):
    """
    Integrand for ParallelMonotonicNN (Independent weights per dimension).
    """
    def __init__(self, num_dims, context_dim, hidden_layers):
        super().__init__()
        self.num_dims = num_dims
        self.context_dim = context_dim
        
        # Each group input: 1 (x_i) + context_dim
        input_per_group = 1 + context_dim
        
        layers = []
        last_h = input_per_group
        
        for h in hidden_layers:
            layers.append(GroupedLinear(num_dims * last_h, num_dims * h, groups=num_dims))
            layers.append(nn.ReLU())
            last_h = h
            
        # Output: 1 per group
        layers.append(GroupedLinear(num_dims * last_h, num_dims * 1, groups=num_dims))
        layers.append(nn.ELU())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, h):
        # x: (Batch, D)
        # h: (Batch, C)
        B, D = x.shape
        C = h.shape[1] if h is not None else 0
        
        if h is not None:
            # We need to construct input: [x1, h, x2, h, ...]
            # h: (B, C) -> (B, D, C)
            h_rep = h.unsqueeze(1).expand(B, D, C)
            # x: (B, D, 1)
            x_rep = x.unsqueeze(2)
            # Concat: (B, D, 1+C)
            inp = torch.cat((x_rep, h_rep), dim=2)
            # Flatten channels: (B, D*(1+C))
            inp = inp.view(B, D * (1 + C))
        else:
            inp = x
            
        return self.net(inp) + 1.

class MonotonicNN(nn.Module):
    """
    Standard Monotonic NN with shared weights across dimensions (if input D > 1).
    """
    def __init__(self, context_dim, hidden_layers, nb_steps=50):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(1 + context_dim, hidden_layers)
        self.nb_steps = nb_steps
        
        # Context net for shared scaling/offset
        hs = [context_dim] + hidden_layers + [2]
        net = []
        for h0, h1 in zip(hs[:-1], hs[1:]):
            net.extend([nn.Linear(h0, h1), nn.ReLU()])
        net.pop()
        self.context_net = nn.Sequential(*net)

    def forward(self, x, h):
        B, D = x.shape
        out = self.context_net(h)
        offset = out[:, 0].view(B, 1)
        scaling = torch.exp(out[:, 1].view(B, 1))

        # Flatten D into Batch for shared processing
        x_flat = x.reshape(-1, 1)
        x0_flat = torch.zeros_like(x_flat)
        
        if h is not None:
            C = h.shape[1]
            h_flat = h.unsqueeze(1).expand(B, D, C).reshape(-1, C)
        else:
            h_flat = None
            
        flat_params = flatten_params(self.integrand.parameters())
        int_val = UMNNNeuralIntegral.apply(x0_flat, x_flat, self.integrand, flat_params, h_flat, self.nb_steps)
        int_val = int_val.view(B, D)
        
        return scaling * int_val + offset

class ParallelMonotonicNN(nn.Module):
    """
    Monotonic NN with independent weights for each input dimension.
    Equivalent to a list of MonotonicNNs but batched.
    """
    def __init__(self, num_dims, context_dim, hidden_layers, nb_steps=50):
        super().__init__()
        self.integrand = ParallelIntegrandNN(num_dims, context_dim, hidden_layers)
        self.nb_steps = nb_steps
        
        hs = [context_dim] + hidden_layers + [2 * num_dims]
        net = []
        for h0, h1 in zip(hs[:-1], hs[1:]):
            net.extend([nn.Linear(h0, h1), nn.ReLU()])
        net.pop()
        self.context_net = nn.Sequential(*net)
        
    def forward(self, x, h):
        B, D = x.shape
        
        # Scaling/Offset
        out = self.context_net(h) # (B, 2*D)
        out = out.view(B, D, 2)
        offset = out[:, :, 0]
        scaling = torch.exp(out[:, :, 1])
        
        x0 = torch.zeros_like(x)
        flat_params = flatten_params(self.integrand.parameters())
        
        int_val = UMNNNeuralIntegral.apply(x0, x, self.integrand, flat_params, h, self.nb_steps)
        # int_val is (B, D)
        
        return scaling * int_val + offset

class GeneralizedUMNN(nn.Module):
    """
    Optimized implementation of SlowDMonotonicNN.
    Maps (B, D) -> (B, 1) using a weighted sum of independent monotonic transformations.
    """
    def __init__(self, num_dims, context_dim, hidden_layers, nb_steps=50):
        super().__init__()
        self.inner_net = ParallelMonotonicNN(num_dims, context_dim, hidden_layers, nb_steps)
        self.weights = nn.Parameter(torch.randn(num_dims))
        # Context net input is just h, so context_dim is enough
        self.outer_net = MonotonicNN(context_dim, hidden_layers, nb_steps)
        
    def forward(self, x, h):
        # x: (B, D), h: (B, C)
        
        # 1. Independent monotonic transformations
        inner_out = self.inner_net(x, h) # (B, D)
        
        # 2. Weighted Sum
        # weights: (D)
        w = torch.exp(self.weights).view(1, -1)
        inner_sum = (inner_out * w).sum(dim=1, keepdim=True) # (B, 1)
        
        # 3. Outer transformation
        out = self.outer_net(inner_sum, h)
        return out