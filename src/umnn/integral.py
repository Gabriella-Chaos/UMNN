import torch
from .utils import compute_cc_weights
from .triton_ops import triton_weighted_sum, triton_backward_expansion, is_triton_available

def flatten_params(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class UMNNNeuralIntegral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20):
        """
        Computes the integral of integrand(t, h) from x0 to x.
        """
        device = x0.device
        dtype = x0.dtype
        
        # 1. Compute Clenshaw-Curtis weights/steps
        cc_weights, steps = compute_cc_weights(nb_steps, device)
        cc_weights = cc_weights.to(dtype)
        steps = steps.to(dtype)

        # 2. Prepare inputs
        # steps: (Steps+1) -> (1, Steps+1, 1)
        steps_expanded = steps.view(1, -1, 1)
        
        # x0, x: (Batch, Dim) -> (Batch, 1, Dim)
        x0_expanded = x0.unsqueeze(1)
        x_expanded = x.unsqueeze(1)
        
        # Linear interpolation
        # (Batch, 1, Dim) + (Batch, 1, Dim) * (1, Steps, 1)
        # Broadcasts to (Batch, Steps, Dim)
        diff = (x_expanded - x0_expanded) / 2.0
        X_steps = x0_expanded + diff * (steps_expanded + 1)
        
        # Flatten
        B, S, D = X_steps.shape
        X_flat = X_steps.reshape(-1, D)
        
        # Expand h
        if h is not None:
            H_D = h.shape[1]
            h_expanded = h.unsqueeze(1).expand(B, S, H_D).reshape(-1, H_D)
        else:
            h_expanded = None
            
        # 3. Evaluate Integrand
        with torch.no_grad():
            out_flat = integrand(X_flat, h_expanded)
            OutD = out_flat.shape[1]
            out = out_flat.view(B, S, OutD)

        # 4. Weighted Sum (Integral)
        if is_triton_available() and x0.is_cuda:
            weighted_sum = triton_weighted_sum(out, cc_weights)
        else:
            w = cc_weights.view(1, -1, 1)
            weighted_sum = torch.sum(out * w, dim=1)
            
        interval_width = (x - x0) / 2.0
        result = weighted_sum * interval_width

        ctx.save_for_backward(x0, x, h, cc_weights, steps)
        ctx.integrand = integrand
        ctx.nb_steps = nb_steps
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, h, cc_weights, steps = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        
        grad_output = grad_output.contiguous()
        
        # 1. Gradients w.r.t x and x0 (Leibniz Rule)
        with torch.enable_grad():
            fx = integrand(x, h)
            fx0 = integrand(x0, h)
            
            grad_x = fx * grad_output
            grad_x0 = -fx0 * grad_output
            
        # 2. Gradients w.r.t parameters and h
        # Recompute inputs
        with torch.no_grad():
            steps_expanded = steps.view(1, -1, 1)
            x0_expanded = x0.unsqueeze(1)
            x_expanded = x.unsqueeze(1)
            diff = (x_expanded - x0_expanded) / 2.0
            X_steps = x0_expanded + diff * (steps_expanded + 1)
            B, S, D = X_steps.shape
            X_flat = X_steps.reshape(-1, D)
            
            if h is not None:
                H_D = h.shape[1]
                h_expanded = h.unsqueeze(1).expand(B, S, H_D).reshape(-1, H_D)
            else:
                h_expanded = None

        # Enable grad for backprop
        if h_expanded is not None and h.requires_grad:
            h_expanded.requires_grad_(True)
            
        # Forward pass for gradient computation
        # We need to track gradients for parameters too
        with torch.enable_grad():
             out_flat = integrand(X_flat, h_expanded)
        
        # Compute gradient target
        out = out_flat.view(B, S, -1) 
        interval_width = (x - x0) / 2.0
        grad_output_scaled = grad_output * interval_width
        
        if is_triton_available() and x0.is_cuda:
            d_out = triton_backward_expansion(grad_output_scaled, cc_weights, out.shape)
        else:
            d_out = grad_output_scaled.unsqueeze(1) * cc_weights.view(1, -1, 1)
            
        # Compute gradients using autograd.grad to avoid accumulating into .grad directly
        
        inputs_to_grad = list(integrand.parameters())
        if h_expanded is not None and h.requires_grad:
            inputs_to_grad.append(h_expanded)
            
        d_out_flat = d_out.reshape(-1, out.shape[2])
        
        # Run backward
        gradients = torch.autograd.grad(out_flat, inputs_to_grad, grad_outputs=d_out_flat)
        
        # Extract parameter grads
        num_params = len(list(integrand.parameters()))
        param_grads = gradients[:num_params]
        grad_flat_params = flatten_params(param_grads)
        
        # Extract h grad
        grad_h = None
        if h_expanded is not None and h.requires_grad:
            grad_h_expanded = gradients[-1]
            grad_h = grad_h_expanded.view(B, S, -1).sum(dim=1)
        
        return grad_x0, grad_x, None, grad_flat_params, grad_h, None
