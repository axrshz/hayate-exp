import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .linear import FeedForward


class TransformerBlock(nn.Module):
    """Run one attention and feed-forward layer with residual connections."""

    def __init__(self, hidden_size=2560, num_heads=32, num_kv_groups=8,
                 head_dim=128, intermediate_size=9728, rms_norm_eps=1e-6,
                 dtype=torch.bfloat16):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_in=hidden_size, num_heads=num_heads,
            num_kv_groups=num_kv_groups, head_dim=head_dim, dtype=dtype,
        )
        self.mlp = FeedForward(emb_dim=hidden_size, hidden_dim=intermediate_size, dtype=dtype)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=dtype)

    def forward(
        self,
        x,
        cos,
        sin,
        position_ids,
        prev_k,
        prev_v,
        varlen_metadata=None,
        paged_cache=None,
        layer_idx=None,
    ):
        # RMSNorm scales hidden features before attention.
        # The residual path adds attention output to the original hidden states.
        shortcut = x
        x = self.input_layernorm(x)
        x, new_k, new_v = self.self_attn(
            x,
            cos,
            sin,
            position_ids,
            prev_k,
            prev_v,
            varlen_metadata,
            paged_cache,
            layer_idx,
        )
        x = x + shortcut

        # RMSNorm scales hidden features before the feed-forward network.
        # The second residual path adds feed-forward output to the attention result.
        shortcut = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + shortcut
        return x, new_k, new_v
