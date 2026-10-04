"""Paged KV storage and FlexAttention metadata for inference requests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from hayate.model.cache import Cache


_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


@dataclass
class PagedCacheHandle:
    """Identify one request's physical KV pages and logical cache length."""

    slot: int
    pages: list[int] = field(default_factory=list)
    length: int = 0
    released: bool = False


class PagedKVBatch(NamedTuple):
    """Tensor metadata shared by every layer for one paged attention call."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    cache_lens: torch.Tensor
    block_mask: BlockMask
    score_mod: object
    page_size: int

    def write_layer(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """Write a dense batch of new key and value states into physical pages."""
        _, _, query_len, _ = k.shape
        logical_positions = self.cache_lens[:, None] + torch.arange(
            query_len, device=k.device, dtype=torch.long
        ).view(1, -1)
        logical_pages = logical_positions // self.page_size
        physical_pages = torch.gather(self.page_table, 1, logical_pages)
        physical_positions = (
            physical_pages * self.page_size + logical_positions % self.page_size
        )

        # The cache has one shared physical sequence; batch rows choose page IDs.
        self.k_cache[layer_idx, 0, :, physical_positions, :] = k.permute(1, 0, 2, 3)
        self.v_cache[layer_idx, 0, :, physical_positions, :] = v.permute(1, 0, 2, 3)

    def attend_layer(self, q: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Attend to this layer's paged cache with FlexAttention."""
        return _compiled_flex_attention(
            q,
            self.k_cache[layer_idx],
            self.v_cache[layer_idx],
            score_mod=self.score_mod,
            block_mask=self.block_mask,
            enable_gqa=True,
        )


class PagedKVCacheManager:
    """Allocate fixed-size pages from shared KV tensors for active requests."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_tokens: int,
        page_size: int,
        max_requests: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        if max_cache_tokens < 1:
            raise ValueError("max_cache_tokens must be positive")
        if page_size < 1 or page_size & (page_size - 1):
            raise ValueError("page_size must be a positive power of two")
        if max_requests < 1:
            raise ValueError("max_requests must be positive")

        self.page_size = page_size
        self.num_pages = math.ceil(max_cache_tokens / page_size)
        self.capacity_tokens = self.num_pages * page_size
        self.max_requests = max_requests
        self.device = torch.device(device)
        self._free_slots = list(range(max_requests - 1, -1, -1))
        self._free_pages = list(range(self.num_pages - 1, -1, -1))

        # One physical pool is shared by every request and every transformer layer.
        self.k_cache = torch.empty(
            (num_layers, 1, num_kv_heads, self.capacity_tokens, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.v_cache = torch.empty_like(self.k_cache)

    def allocate(self) -> PagedCacheHandle:
        """Reserve a request slot without allocating KV pages yet."""
        if not self._free_slots:
            raise MemoryError(
                f"paged cache supports at most {self.max_requests} active requests"
            )
        return PagedCacheHandle(slot=self._free_slots.pop())

    def reserve(self, handle: PagedCacheHandle, required_tokens: int) -> None:
        """Allocate enough pages for a request's logical token positions."""
        if handle.released:
            raise RuntimeError("cannot reserve pages for a released cache handle")
        if required_tokens < 0:
            raise ValueError("required_tokens must be non-negative")
        if required_tokens > self.capacity_tokens:
            raise MemoryError(
                f"request needs {required_tokens} cache tokens, but the paged cache "
                f"capacity is {self.capacity_tokens} tokens"
            )

        required_pages = math.ceil(required_tokens / self.page_size)
        missing_pages = required_pages - len(handle.pages)
        if missing_pages <= 0:
            return
        if len(self._free_pages) < missing_pages:
            raise MemoryError(
                f"paged KV cache is full: need {missing_pages} additional pages, "
                f"but only {len(self._free_pages)} remain"
            )
        handle.pages.extend(self._free_pages.pop() for _ in range(missing_pages))

    def allocate_for_request(self, cache: Cache | None = None) -> PagedCacheHandle:
        """Allocate a request handle and optionally import a contiguous prefix."""
        handle = self.allocate()
        try:
            if cache is not None and cache.length:
                self.import_cache(handle, cache)
        except Exception:
            self.release(handle)
            raise
        return handle

    def import_cache(self, handle: PagedCacheHandle, cache: Cache) -> None:
        """Copy a contiguous prefix-cache entry into the request's physical pages."""
        if cache.k is None or cache.v is None:
            return
        self.reserve(handle, cache.length)
        positions = self._physical_positions(handle, cache.length)
        for layer_idx in range(cache.k.shape[0]):
            self.k_cache[layer_idx, 0].index_copy_(
                1, positions, cache.k[layer_idx, :, : cache.length, :]
            )
            self.v_cache[layer_idx, 0].index_copy_(
                1, positions, cache.v[layer_idx, :, : cache.length, :]
            )
        handle.length = cache.length

    def truncate(self, handle: PagedCacheHandle, length: int) -> None:
        """Shorten a cache and return whole pages beyond its new end to the pool."""
        if handle.released:
            raise RuntimeError("cannot truncate a released cache handle")
        if length < 0 or length > handle.length:
            raise ValueError("truncate length must be within the current cache length")
        keep_pages = math.ceil(length / self.page_size)
        released = handle.pages[keep_pages:]
        handle.pages = handle.pages[:keep_pages]
        self._free_pages.extend(released)
        handle.length = length

    def snapshot(self, handle: PagedCacheHandle, length: int | None = None) -> Cache:
        """Materialize a compact cache snapshot for the optional prefix cache."""
        if handle.released:
            raise RuntimeError("cannot snapshot a released cache handle")
        snapshot_len = handle.length if length is None else length
        if snapshot_len < 0 or snapshot_len > handle.length:
            raise ValueError("snapshot length must be within the current cache length")
        positions = self._physical_positions(handle, snapshot_len)
        k = self.k_cache[:, 0].index_select(2, positions).contiguous()
        v = self.v_cache[:, 0].index_select(2, positions).contiguous()
        return Cache.from_tensors(k, v)

    def release(self, handle: PagedCacheHandle) -> None:
        """Return all of a request's pages and its slot to the free pools."""
        if handle.released:
            return
        self._free_pages.extend(handle.pages)
        handle.pages.clear()
        handle.length = 0
        handle.released = True
        self._free_slots.append(handle.slot)

    def begin_batch(
        self, handles: Sequence[PagedCacheHandle], query_len: int
    ) -> PagedKVBatch:
        """Reserve append space and build a page-aware causal FlexAttention mask."""
        if not handles:
            raise ValueError("cannot build paged cache metadata for an empty batch")
        if query_len < 1:
            raise ValueError("query_len must be positive")
        if any(handle.released for handle in handles):
            raise RuntimeError("cannot use a released cache handle in an attention batch")
        if len({id(handle) for handle in handles}) != len(handles):
            raise ValueError("an attention batch cannot contain the same cache handle twice")

        batch_size = len(handles)
        lengths = [handle.length for handle in handles]
        kv_lengths = [length + query_len for length in lengths]
        required_pages = [
            math.ceil(length / self.page_size) for length in kv_lengths
        ]
        additional_pages = sum(
            max(0, required - len(handle.pages))
            for handle, required in zip(handles, required_pages)
        )
        if any(length > self.capacity_tokens for length in kv_lengths):
            raise MemoryError(
                f"request needs more than the paged cache capacity of "
                f"{self.capacity_tokens} tokens"
            )
        if additional_pages > len(self._free_pages):
            raise MemoryError(
                f"paged KV cache is full: batch needs {additional_pages} "
                f"additional pages, but only {len(self._free_pages)} remain"
            )
        for handle, target_length in zip(handles, kv_lengths):
            self.reserve(handle, target_length)

        page_table = torch.full(
            (batch_size, self.num_pages),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        physical_to_logical = torch.full_like(page_table, -1)
        for row, handle in enumerate(handles):
            count = len(handle.pages)
            page_ids = torch.tensor(
                handle.pages, dtype=torch.long, device=self.device
            )
            page_table[row, :count] = page_ids
            physical_to_logical[row, page_ids] = torch.arange(
                count, dtype=torch.long, device=self.device
            )

        cache_lens = torch.tensor(lengths, dtype=torch.long, device=self.device)
        kv_lens = torch.tensor(kv_lengths, dtype=torch.long, device=self.device)

        def causal_mask(batch, _head, query_idx, key_idx):
            return (
                (key_idx >= 0)
                & (key_idx < kv_lens[batch])
                & (key_idx <= cache_lens[batch] + query_idx)
            )

        logical_mask = create_block_mask(
            causal_mask,
            B=batch_size,
            H=None,
            Q_LEN=query_len,
            KV_LEN=max(kv_lengths),
            device=self.device,
            BLOCK_SIZE=(self.page_size, self.page_size),
        )
        physical_mask, score_mod = self._convert_block_mask(
            logical_mask, page_table, physical_to_logical, kv_lens
        )
        return PagedKVBatch(
            self.k_cache,
            self.v_cache,
            page_table,
            cache_lens,
            physical_mask,
            score_mod,
            self.page_size,
        )

    def commit_batch(
        self, handles: Sequence[PagedCacheHandle], query_len: int
    ) -> None:
        """Advance logical lengths after every layer has written the new states."""
        for handle in handles:
            if handle.released:
                raise RuntimeError("cannot commit a released cache handle")
            if handle.length + query_len > len(handle.pages) * self.page_size:
                raise RuntimeError("paged cache append space was not reserved")
            handle.length += query_len

    def _physical_positions(
        self, handle: PagedCacheHandle, length: int
    ) -> torch.Tensor:
        if length == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        logical = torch.arange(length, dtype=torch.long, device=self.device)
        page_ids = torch.tensor(handle.pages, dtype=torch.long, device=self.device)
        return page_ids[logical // self.page_size] * self.page_size + (
            logical % self.page_size
        )

    def _convert_block_mask(
        self,
        block_mask: BlockMask,
        page_table: torch.Tensor,
        physical_to_logical: torch.Tensor,
        kv_lens: torch.Tensor,
    ) -> tuple[BlockMask, object]:
        """Translate logical KV blocks to physical pages for FlexAttention."""
        batch_size, heads, rows, max_blocks = block_mask.kv_indices.shape
        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise ValueError("FlexAttention KV block size must equal the KV page size")

        mapped_indices = torch.gather(
            page_table,
            1,
            block_mask.kv_indices.reshape(batch_size, -1).to(torch.long),
        ).reshape(block_mask.kv_indices.shape)
        new_kv_indices = torch.zeros(
            (batch_size, heads, rows, self.num_pages),
            dtype=torch.int32,
            device=self.device,
        )
        new_kv_indices[:, :, :, :max_blocks] = mapped_indices.to(torch.int32)
        new_full_indices = None
        new_full_counts = None
        if block_mask.full_kv_num_blocks is not None:
            assert block_mask.full_kv_indices is not None
            full_shape = block_mask.full_kv_indices.shape
            mapped_full = torch.gather(
                page_table,
                1,
                block_mask.full_kv_indices.reshape(batch_size, -1).to(torch.long),
            ).reshape(full_shape)
            new_full_indices = torch.zeros(
                (batch_size, heads, rows, self.num_pages),
                dtype=torch.int32,
                device=self.device,
            )
            new_full_indices[:, :, :, : full_shape[-1]] = mapped_full.to(torch.int32)
            new_full_counts = block_mask.full_kv_num_blocks.clone()

        def logical_positions(batch, physical_key):
            physical_page = physical_key // self.page_size
            page_offset = physical_key % self.page_size
            logical_page = physical_to_logical[batch, physical_page]
            logical_key = logical_page * self.page_size + page_offset
            valid = (logical_page >= 0) & (logical_key >= 0) & (
                logical_key < kv_lens[batch]
            )
            return logical_key, valid

        def converted_mask(batch, head, query_idx, physical_key):
            logical_key, valid = logical_positions(batch, physical_key)
            return valid & block_mask.mask_mod(batch, head, query_idx, logical_key)

        def converted_score(score, batch, head, query_idx, physical_key):
            _, valid = logical_positions(batch, physical_key)
            return torch.where(valid, score, float("-inf"))

        physical_mask = BlockMask.from_kv_blocks(
            block_mask.kv_num_blocks.clone(),
            new_kv_indices,
            new_full_counts,
            new_full_indices,
            block_mask.BLOCK_SIZE,
            converted_mask,
            seq_lengths=(block_mask.seq_lengths[0], self.capacity_tokens),
        )
        return physical_mask, converted_score
