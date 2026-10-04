import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope_vectorized

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.varlen import varlen_attn


def _flash_sdpa_context():
    """Allow Flash Attention and keep PyTorch's math kernel as a fallback."""
    # Hayate uses the math kernel when a valid input cannot use Flash Attention.
    return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])


class GroupedQueryAttention(nn.Module):
    """Compute grouped-query attention and return the new key and value states."""

    def __init__(self, d_in, num_heads, num_kv_groups, head_dim, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups

        self.q_proj = nn.Linear(d_in, num_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_heads * head_dim, d_in, bias=False, dtype=dtype)

        # Normalize each query and key head before position rotation.
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6, dtype=dtype)

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
        """Apply attention with optional history and variable-length metadata.

        x has shape (batch, tokens, hidden size).
        Each cache has shape (batch, KV heads, history, head size).
        The method returns output values and key and value states for new tokens.
        """
        B, T, _ = x.shape

        # Separate projections create query heads and the smaller set of KV heads.
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention kernels expect heads before the token axis.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope_vectorized(q, cos, sin, position_ids)
        k = apply_rope_vectorized(k, cos, sin, position_ids)

        if paged_cache is not None:
            # Store the current tokens, then attend through the physical page table.
            paged_cache.write_layer(layer_idx, k, v)
            context = paged_cache.attend_layer(q, layer_idx).transpose(1, 2)
        else:
            # Join new states to prior states so each query can use its full history.
            if prev_k is not None:
                full_k = torch.cat([prev_k, k], dim=2)
                full_v = torch.cat([prev_v, v], dim=2)
            else:
                full_k, full_v = k, v

            if varlen_metadata is not None:
                # Pack only valid positions so the kernel does not compute on padding.
                valid_q, valid_k, cu_q, cu_k, max_q, max_k, is_causal = varlen_metadata
                q_packed = q.transpose(1, 2)[valid_q]
                k_packed = full_k.transpose(1, 2)[valid_k]
                v_packed = full_v.transpose(1, 2)[valid_k]
                packed_context = varlen_attn(
                    q_packed,
                    k_packed,
                    v_packed,
                    cu_q,
                    cu_k,
                    max_q,
                    max_k,
                    is_causal=is_causal,
                )
                context = torch.zeros(
                    B, T, self.num_heads, self.head_dim, dtype=x.dtype, device=x.device
                )
                # Restore each packed result to its original batch and token position.
                context[valid_q] = packed_context
            else:
                # Dense batches need no padding mask, so PyTorch can use Flash Attention.
                with _flash_sdpa_context():
                    context = F.scaled_dot_product_attention(
                        q,
                        full_k,
                        full_v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=T > 1,
                        enable_gqa=True,
                    ).transpose(1, 2)
        context = context.reshape(B, T, self.num_heads * self.head_dim)
        # Return new states only because the engine already stores the old history.
        return self.o_proj(context), (None if paged_cache is not None else k), (
            None if paged_cache is not None else v
        )
