import torch
import torch.nn as nn

from .rope import compute_rope_params
from .block import TransformerBlock


class Qwen3Model(nn.Module):
    """Define the Qwen3 4B transformer that Hayate runs during inference."""

    def __init__(self):
        super().__init__()

        # The vocabulary size sets the embedding table and the number of output scores.
        vocab_size = 151_936
        # Hidden size sets the number of features that represent each token.
        hidden_size = 2560
        # The model applies this many transformer blocks in sequence.
        num_layers = 36
        # Query heads share a smaller set of key and value heads.
        num_heads = 32
        num_kv_groups = 8
        # Each attention head uses this many features.
        head_dim = 128
        # The feed-forward network expands each token to this many intermediate features.
        intermediate_size = 9728
        # RMSNorm adds this small value to keep normalization stable near zero.
        rms_norm_eps = 1e-6
        # RoPE settings define token positions supported by this checkpoint.
        rope_theta = 1_000_000
        max_position_embeddings = 40_960

        self.num_layers = num_layers
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # BF16 model weights use less GPU memory than FP32 weights.
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size, num_heads=num_heads,
                num_kv_groups=num_kv_groups, head_dim=head_dim,
                intermediate_size=intermediate_size, rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps, dtype=torch.bfloat16)
        self.out_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)

        # RoPE tables do not belong in the checkpoint because Hayate can rebuild them.
        cos, sin = compute_rope_params(
            head_dim=head_dim, theta_base=rope_theta,
            context_length=max_position_embeddings
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def initialize_rope(self, device: torch.device | str) -> None:
        """Build the position tables on the device that stores model weights."""
        self.cos, self.sin = compute_rope_params(
            head_dim=self.head_dim,
            theta_base=self.rope_theta,
            context_length=self.max_position_embeddings,
            device=device,
        )

    def forward(
        self,
        token_ids,
        prev_k=None,
        prev_v=None,
        cache_lens=None,
        pad_lengths=None,
        use_varlen=False,
        paged_cache=None,
    ):
        """Run all transformer layers and return next-token scores with new cache states.

        token_ids has shape (batch, tokens).
        Each prior cache has shape (layers, batch, KV heads, history, head size).
        Scores have shape (batch, 1, vocabulary size).
        New key and value states have shape (layers, batch, KV heads, tokens, head size).
        """
        B, T = token_ids.shape
        x = self.embed_tokens(token_ids)

        if paged_cache is not None:
            cache_lens = paged_cache.cache_lens

        # The batched cache uses the longest history; cache_lens marks each real length.
        L_prev = prev_k.shape[3] if prev_k is not None else 0
        L_full = L_prev + T

        # Start with positions inside this new token block.
        arange_t = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0)

        if cache_lens is not None:
            cl_1d = cache_lens.view(-1, 1)
        else:
            cl_1d = torch.zeros(B, 1, device=x.device, dtype=torch.long)

        if pad_lengths is not None:
            # Subtract left padding so real tokens keep their correct absolute positions.
            pl_1d = pad_lengths.view(-1, 1)
            position_ids = (arange_t + cl_1d - pl_1d).clamp(min=0)
        else:
            position_ids = arange_t + cl_1d

        varlen_metadata = None
        if use_varlen:
            # Build token masks and offsets that describe each request's valid sequence.
            if pad_lengths is None:
                pad_lengths = torch.zeros(B, device=x.device, dtype=torch.long)
            query_lens = T - pad_lengths
            if cache_lens is None:
                cache_lens = torch.zeros(B, device=x.device, dtype=torch.long)
            key_lens = cache_lens + query_lens

            # Cumulative lengths mark where each request starts in packed tensors.
            cu_q = torch.zeros(B + 1, device=x.device, dtype=torch.int32)
            cu_k = torch.zeros(B + 1, device=x.device, dtype=torch.int32)
            cu_q[1:] = query_lens.cumsum(0).to(torch.int32)
            cu_k[1:] = key_lens.cumsum(0).to(torch.int32)

            # Exclude left-pad queries and padded cache positions from attention.
            valid_q = arange_t >= pad_lengths.view(-1, 1)
            cache_pos = torch.arange(L_prev, device=x.device).view(1, -1)
            valid_cache = cache_pos < cache_lens.view(-1, 1)
            valid_k = torch.cat([valid_cache, valid_q], dim=1)
            varlen_metadata = (valid_q, valid_k, cu_q, cu_k, T, L_full, T > 1)

        # Each layer reads only its matching slice of the prior cache.
        new_k_list = []
        new_v_list = []
        for layer_idx, block in enumerate(self.layers):
            pk = prev_k[layer_idx] if prev_k is not None else None
            pv = prev_v[layer_idx] if prev_v is not None else None
            x, nk, nv = block(
                x,
                self.cos,
                self.sin,
                position_ids,
                pk,
                pv,
                varlen_metadata,
                paged_cache,
                layer_idx,
            )
            if paged_cache is None:
                new_k_list.append(nk)
                new_v_list.append(nv)

        # Contiguous callers receive cache states; the engine uses paged storage.
        if paged_cache is None:
            new_k = torch.stack(new_k_list, dim=0)
            new_v = torch.stack(new_v_list, dim=0)
        else:
            new_k = new_v = None

        x = self.norm(x)
        # The final token predicts the next output token, so earlier logits are not needed.
        logits = self.out_head(x[:, -1:, :])
        return logits, new_k, new_v
