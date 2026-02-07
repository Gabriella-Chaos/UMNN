import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

def is_triton_available():
    return triton is not None

if is_triton_available():
    @triton.jit
    def weighted_reduction_kernel(
        x_ptr, w_ptr, out_ptr,
        n_batch, n_steps, n_dim,
        stride_xb, stride_xs, stride_xd,
        stride_w,
        stride_ob, stride_od,
        BLOCK_SIZE_S: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        
        # Offsets for batch and dimension
        offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        
        # Accumulator
        acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
        
        # Iterate over steps
        for s in range(0, n_steps, BLOCK_SIZE_S):
            offs_s = s + tl.arange(0, BLOCK_SIZE_S)
            mask_s = offs_s < n_steps
            mask_d = offs_d < n_dim
            
            # Load weights
            w = tl.load(w_ptr + offs_s * stride_w, mask=mask_s, other=0.0)
            
            # Load x
            # Pointer arithmetic: batch_offset + step_offset + dim_offset
            x_ptrs = x_ptr + (pid_b * stride_xb) + (offs_s[:, None] * stride_xs) + (offs_d[None, :] * stride_xd)
            x = tl.load(x_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
            
            # Weighted sum
            acc += tl.sum(x * w[:, None], axis=0)
            
        # Store result
        out_ptrs = out_ptr + (pid_b * stride_ob) + (offs_d * stride_od)
        mask_out = offs_d < n_dim
        tl.store(out_ptrs, acc, mask=mask_out)

    @triton.jit
    def backward_weighted_expansion_kernel(
        grad_out_ptr, w_ptr, grad_x_ptr,
        n_batch, n_steps, n_dim,
        stride_gob, stride_god,
        stride_w,
        stride_gxb, stride_gxs, stride_gxd,
        BLOCK_SIZE_S: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr
    ):
        """
        Computes grad_x = grad_out * w
        grad_out: (Batch, Dim)
        w: (Steps)
        grad_x: (Batch, Steps, Dim)
        """
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1) # Block over steps
        pid_d = tl.program_id(2) # Block over dims

        offs_s = pid_s * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
        offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)

        mask_s = offs_s < n_steps
        mask_d = offs_d < n_dim

        # Load grad_out
        # Shape (Batch, Dim)
        go_ptrs = grad_out_ptr + (pid_b * stride_gob) + (offs_d * stride_god)
        grad_out = tl.load(go_ptrs, mask=mask_d, other=0.0)

        # Load weights
        # Shape (Steps)
        w_ptrs = w_ptr + (offs_s * stride_w)
        w = tl.load(w_ptrs, mask=mask_s, other=0.0)

        # Compute grad_x = grad_out * w (Broadcasting)
        # grad_out: (BLOCK_D)
        # w: (BLOCK_S)
        # result: (BLOCK_S, BLOCK_D)
        val = grad_out[None, :] * w[:, None]

        # Store grad_x
        # Shape (Batch, Steps, Dim)
        gx_ptrs = grad_x_ptr + (pid_b * stride_gxb) + (offs_s[:, None] * stride_gxs) + (offs_d[None, :] * stride_gxd)
        mask_gx = mask_s[:, None] & mask_d[None, :]
        tl.store(gx_ptrs, val, mask=mask_gx)


def triton_weighted_sum(x, weights):
    """
    x: (Batch, Steps, Dim)
    weights: (Steps)
    Returns: (Batch, Dim)
    """
    if not is_triton_available():
        raise RuntimeError("Triton is not available")
        
    B, S, D = x.shape
    out = torch.empty((B, D), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (B, triton.cdiv(D, META['BLOCK_SIZE_D']))
    
    weighted_reduction_kernel[grid](
        x, weights, out,
        B, S, D,
        x.stride(0), x.stride(1), x.stride(2),
        weights.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_S=32, # Tune these
        BLOCK_SIZE_D=64
    )
    return out

def triton_backward_expansion(grad_out, weights, x_shape):
    """
    grad_out: (Batch, Dim)
    weights: (Steps)
    x_shape: (Batch, Steps, Dim)
    Returns: (Batch, Steps, Dim)
    """
    if not is_triton_available():
        raise RuntimeError("Triton is not available")
        
    B, S, D = x_shape
    grad_x = torch.empty(x_shape, device=grad_out.device, dtype=grad_out.dtype)
    
    # Grid: (Batch, Steps_Blocks, Dim_Blocks)
    grid = lambda META: (B, triton.cdiv(S, META['BLOCK_SIZE_S']), triton.cdiv(D, META['BLOCK_SIZE_D']))
    
    backward_weighted_expansion_kernel[grid](
        grad_out, weights, grad_x,
        B, S, D,
        grad_out.stride(0), grad_out.stride(1),
        weights.stride(0),
        grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
        BLOCK_SIZE_S=32,
        BLOCK_SIZE_D=64
    )
    return grad_x
