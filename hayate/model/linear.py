import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Apply the gated feed-forward network used by each transformer block."""

    def __init__(self, emb_dim: int = 2048, hidden_dim: int = 8192, dtype: torch.dtype = torch.bfloat16):
        """Create the learned projections between hidden and intermediate sizes."""
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.up_proj   = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor):
        """Transform each token's hidden features with the gated network."""
        # The gate and value projections use separate learned weight matrices.
        x_fc1 = self.gate_proj(x)
        x_fc2 = self.up_proj(x)
        # SiLU gates the value features before the final projection returns hidden size.
        x = nn.functional.silu(x_fc1) * x_fc2 
        return self.down_proj(x)
