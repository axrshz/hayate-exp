from typing import List

import torch

from hayate.engine.request import Request


def gather_caches(requests: List[Request], num_layers: int):
    """Build batched key and value tensors from the request caches.

    Each request cache has shape (layers, KV heads, tokens, head size).
    The function returns batched tensors and the valid history length for each request.
    """
    batch_size = len(requests)
    cache_lens_py: List[int] = []
    ref: torch.Tensor | None = None
    for r in requests:
        cache = r.kv_cache
        if cache is not None and cache.k is not None:
            cache_lens_py.append(cache.length)
            if ref is None:
                ref = cache.k
        else:
            cache_lens_py.append(0)

    # The model accepts missing caches when every request is new.
    if ref is None:
        return None, None, None, cache_lens_py

    # A single request can use cache views without allocating a padded batch.
    if batch_size == 1:
        cache = requests[0].kv_cache
        assert cache is not None and cache.k is not None and cache.v is not None
        length = cache.length
        cache_lens = torch.tensor([length], dtype=torch.long, device=cache.k.device)
        return (
            cache.k[:, :, :length, :].unsqueeze(1),
            cache.v[:, :, :length, :].unsqueeze(1),
            cache_lens,
            cache_lens_py,
        )

    # A batch needs equal tensor lengths, so pad shorter histories with zeros.
    # cache_lens tells attention which positions contain real history.
    max_cache_len = max(cache_lens_py)
    _, H_kv, _, D = ref.shape
    dtype, dev = ref.dtype, ref.device

    prev_k = torch.zeros(num_layers, batch_size, H_kv, max_cache_len, D, dtype=dtype, device=dev)
    prev_v = torch.zeros(num_layers, batch_size, H_kv, max_cache_len, D, dtype=dtype, device=dev)

    for i, r in enumerate(requests):
        L_i = cache_lens_py[i]
        if L_i > 0 and r.kv_cache is not None and r.kv_cache.k is not None:
            prev_k[:, i, :, :L_i, :] = r.kv_cache.k[:, :, :L_i, :]
            prev_v[:, i, :, :L_i, :] = r.kv_cache.v[:, :, :L_i, :]

    cache_lens = torch.tensor(cache_lens_py, dtype=torch.long, device=dev)
    return prev_k, prev_v, cache_lens, cache_lens_py


def scatter_caches(
    requests: List[Request],
    new_k,
    new_v,
    num_tokens: int,
    pad_lengths_py: List[int] | None = None,
):
    """Append each request's new key and value states to its own cache.

    New tensors have shape (layers, batch, KV heads, tokens, head size).
    """
    for i, r in enumerate(requests):
        if r.kv_cache is None:
            continue
        # Skip left-padding tokens because they are not part of this request.
        pl = pad_lengths_py[i] if pad_lengths_py is not None else 0
        k_new = new_k[:, i, :, pl:num_tokens, :]
        v_new = new_v[:, i, :, pl:num_tokens, :]
        # The final sampled token needs no cache entry because generation stops.
        capacity = len(r.prompt_tokens) + r.max_tokens - 1
        r.kv_cache.append(k_new, v_new, capacity=capacity)
