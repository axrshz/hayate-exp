from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from hayate.model.cache import Cache


@dataclass(frozen=True)
class PrefixCacheMatch:
    """Describe the cached token count and cache found for one prompt."""

    length: int
    cache: Cache | None


class PrefixCache:
    """Reuse prompt key and value states with a least-recently-used cache.

    The cache removes entries with the oldest use when storage exceeds its budget.
    """

    def __init__(self, max_tokens: int):
        """Set a storage budget measured as the total number of cached tokens."""
        if max_tokens < 0:
            raise ValueError("prefix cache token budget must be non-negative")
        self.max_tokens = max_tokens
        self.total_tokens = 0
        self._entries: OrderedDict[tuple[int, ...], Cache] = OrderedDict()

    def get(self, tokens: Sequence[int], max_prefix_len: int | None = None) -> PrefixCacheMatch:
        """Return the longest cached prefix that matches the supplied token IDs."""
        if self.max_tokens == 0 or not tokens or not self._entries:
            return PrefixCacheMatch(0, None)

        token_tuple = tuple(tokens)
        limit = len(token_tuple) if max_prefix_len is None else min(len(token_tuple), max_prefix_len)
        if limit <= 0:
            return PrefixCacheMatch(0, None)

        best_key: tuple[int, ...] | None = None
        best_len = 0

        # Check every cached key because a longer key can share a useful prefix.
        for cached_tokens, cache in self._entries.items():
            common_len = self._common_prefix_len(token_tuple, cached_tokens, limit)
            if common_len > best_len and common_len <= cache.length:
                best_key = cached_tokens
                best_len = common_len
                if best_len == limit:
                    break

        if best_key is None:
            return PrefixCacheMatch(0, None)

        # A recent hit becomes newer in the LRU order and leaves eviction last.
        self._entries.move_to_end(best_key)
        return PrefixCacheMatch(best_len, self._entries[best_key].slice(best_len))

    def put(self, tokens: Sequence[int], cache: Cache) -> None:
        """Store a compact snapshot of the prompt states within the token budget."""
        if self.max_tokens == 0 or not tokens or cache.length == 0:
            return

        store_len = min(len(tokens), cache.length, self.max_tokens)
        if store_len <= 0:
            return

        key = tuple(tokens[:store_len])
        # Active request caches can change, so prefix entries need independent storage.
        entry = cache.slice(store_len, clone=True)

        existing = self._entries.pop(key, None)
        if existing is not None:
            self.total_tokens -= len(key)

        self._entries[key] = entry
        self.total_tokens += len(key)
        self._evict_to_budget()

    def clear(self) -> None:
        """Remove all entries and reset the token count."""
        self._entries.clear()
        self.total_tokens = 0

    @staticmethod
    def _common_prefix_len(left: Sequence[int], right: Sequence[int], limit: int) -> int:
        """Count equal token IDs from the start of two sequences."""
        count = 0
        max_len = min(len(left), len(right), limit)
        while count < max_len and left[count] == right[count]:
            count += 1
        return count

    def _evict_to_budget(self) -> None:
        """Remove the least recently used entries until storage fits the budget."""
        while self.total_tokens > self.max_tokens and self._entries:
            tokens, _ = self._entries.popitem(last=False)
            self.total_tokens -= len(tokens)
