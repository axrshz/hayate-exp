import unittest

import torch

from hayate.engine.prefix_cache import PrefixCache
from hayate.model.cache import Cache


class CacheTests(unittest.TestCase):
    """Check cache growth and safe reuse of stored prompt states."""

    def test_append_reserves_and_preserves_values(self):
        # Appending must preserve old values and honor the requested capacity.
        cache = Cache()
        first = torch.arange(24).reshape(1, 2, 3, 4)
        final = torch.full((1, 2, 1, 4), 99)

        cache.append(first, first, capacity=8)
        cache.append(final, final)

        self.assertEqual(cache.length, 4)
        self.assertEqual(cache.capacity, 8)
        torch.testing.assert_close(cache.k[:, :, :3], first)
        torch.testing.assert_close(cache.k[:, :, 3:4], final)

    def test_prefix_cache_owns_immutable_compact_storage(self):
        # A prefix entry must not change when its active request cache changes.
        active = Cache()
        values = torch.arange(24).reshape(1, 2, 3, 4)
        active.append(values, values, capacity=8)
        prefixes = PrefixCache(max_tokens=16)

        prefixes.put([1, 2, 3], active)
        active.k[:, :, :3].zero_()
        match = prefixes.get([1, 2, 3, 4])

        self.assertEqual(match.length, 3)
        self.assertEqual(match.cache.capacity, 3)
        torch.testing.assert_close(match.cache.k, values)


if __name__ == "__main__":
    unittest.main()
