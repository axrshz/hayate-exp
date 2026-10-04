import torch


def compute_rope_params(
    head_dim: int,
    theta_base: int = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    """Create cosine and sine tables for rotary position encoding (RoPE)."""
    # RoPE rotates pairs of features, so the head dimension must be even.
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Each feature pair rotates at a different frequency.
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device)[: head_dim // 2].float() / head_dim)
    )
    positions = torch.arange(context_length, dtype=dtype, device=device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) 
    # The model layout uses two halves of the head for each rotation pair.
    angles = torch.cat([angles, angles], dim=1)

    # The returned tables have shape (context_length, head_dim).
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope_vectorized(x, cos, sin, position_ids):
    """Rotate query or key features by each token's absolute position."""
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split the final feature axis into halves and rotate them as pairs.
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Add a head axis so one position table can broadcast across all heads.
    cos_selected = cos[position_ids].unsqueeze(1)
    sin_selected = sin[position_ids].unsqueeze(1)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_selected) + (rotated * sin_selected)

    return x_rotated.to(dtype=x.dtype)
