from __future__ import annotations

import torch


class Cache:
    """Store one request's attention keys and values for every processed token.

    Each tensor has shape (layers, KV heads, tokens, head size).
    """

    __slots__ = ("k", "v", "_length")

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self._length = 0

    @classmethod
    def from_tensors(cls, k: torch.Tensor | None, v: torch.Tensor | None) -> "Cache":
        """Create a cache from tensors with shape (layers, KV heads, tokens, features)."""
        cache = cls()
        cache.k = k
        cache.v = v
        cache._length = 0 if k is None else k.shape[2]
        return cache

    @property
    def length(self) -> int:
        """Return the number of valid token positions in the cache."""
        return self._length

    @property
    def capacity(self) -> int:
        """Return the number of token positions allocated in cache storage."""
        return 0 if self.k is None else self.k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor, capacity: int | None = None) -> None:
        """Append new key and value states, and grow storage when required."""
        if k.shape != v.shape:
            raise ValueError("key and value cache tensors must have matching shapes")
        new_tokens = k.shape[2]
        if new_tokens == 0:
            return

        # Allocate extra space so later decode steps do not copy the full cache.
        required = self._length + new_tokens
        if self.k is None or self.v is None or self.capacity < required:
            new_capacity = max(required, capacity or 0, max(1, self.capacity * 2))
            shape = (*k.shape[:2], new_capacity, k.shape[3])
            new_k = torch.empty(shape, dtype=k.dtype, device=k.device)
            new_v = torch.empty(shape, dtype=v.dtype, device=v.device)
            # Copy valid history into the larger storage before replacing the tensors.
            if self.k is not None and self.v is not None and self._length:
                new_k[:, :, : self._length, :].copy_(self.k[:, :, : self._length, :])
                new_v[:, :, : self._length, :].copy_(self.v[:, :, : self._length, :])
            self.k, self.v = new_k, new_v

        self.k[:, :, self._length : required, :].copy_(k)
        self.v[:, :, self._length : required, :].copy_(v)
        self._length = required

    def slice(self, end: int, clone: bool = False) -> "Cache":
        """Return the first end positions, with an optional independent copy."""
        if end < 0:
            raise ValueError("cache slice end must be non-negative")
        if self.k is None or self.v is None:
            return Cache()
        if end > self.length:
            raise ValueError(f"cache slice end {end} exceeds cache length {self.length}")

        k = self.k[:, :, :end, :]
        v = self.v[:, :, :end, :]
        if clone:
            k = k.contiguous().clone()
            v = v.contiguous().clone()
        return Cache.from_tensors(k, v)

    def reset(self) -> None:
        """Release key and value tensors and mark the cache as empty."""
        self.k = None
        self.v = None
        self._length = 0
