import unittest

import torch
import torch.nn.functional as F

from hayate.engine.paged_cache import PagedKVCacheManager
from hayate.model.cache import Cache


class PagedCacheManagerTests(unittest.TestCase):
    """Check page allocation, cache snapshots, and slot reuse on CPU."""

    def make_manager(self, max_cache_tokens=8):
        return PagedKVCacheManager(
            num_layers=2,
            num_kv_heads=1,
            head_dim=4,
            max_cache_tokens=max_cache_tokens,
            page_size=2,
            max_requests=2,
            dtype=torch.float32,
            device="cpu",
        )

    def test_import_snapshot_and_release_reuse_pages(self):
        manager = self.make_manager()
        values = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
        cache = Cache.from_tensors(values, values + 1)

        handle = manager.allocate_for_request(cache)
        original_pages = list(handle.pages)
        snapshot = manager.snapshot(handle)

        self.assertEqual(handle.length, 3)
        self.assertEqual(len(original_pages), 2)
        torch.testing.assert_close(snapshot.k, values)
        torch.testing.assert_close(snapshot.v, values + 1)

        manager.release(handle)
        replacement = manager.allocate()
        manager.reserve(replacement, 3)
        self.assertEqual(set(replacement.pages), set(original_pages))

    def test_pool_reports_exhausted_pages(self):
        manager = self.make_manager(max_cache_tokens=4)
        first = manager.allocate()
        second = manager.allocate()
        manager.reserve(first, 4)

        with self.assertRaises(MemoryError):
            manager.reserve(second, 1)


@unittest.skipUnless(torch.cuda.is_available(), "FlexAttention paged path requires CUDA")
class FlexPagedAttentionTests(unittest.TestCase):
    """Compare page-table attention with dense GQA over the same logical cache."""

    def test_paged_gqa_matches_dense_attention_for_uneven_histories(self):
        torch.manual_seed(23)
        dtype = torch.bfloat16
        manager = PagedKVCacheManager(
            num_layers=1,
            num_kv_heads=2,
            head_dim=16,
            max_cache_tokens=32,
            page_size=4,
            max_requests=2,
            dtype=dtype,
            device="cuda",
        )
        old_lengths = (3, 5)
        handles = []
        old_keys = []
        old_values = []
        for length in old_lengths:
            keys = torch.randn(1, 2, length, 16, device="cuda", dtype=dtype)
            values = torch.randn_like(keys)
            handles.append(
                manager.allocate_for_request(Cache.from_tensors(keys, values))
            )
            old_keys.append(keys[0])
            old_values.append(values[0])

        query_len = 2
        state = manager.begin_batch(handles, query_len)
        q = torch.randn(2, 4, query_len, 16, device="cuda", dtype=dtype)
        new_k = torch.randn(2, 2, query_len, 16, device="cuda", dtype=dtype)
        new_v = torch.randn_like(new_k)

        with torch.inference_mode():
            state.write_layer(0, new_k, new_v)
            actual = state.attend_layer(q, 0)

        for row, old_length in enumerate(old_lengths):
            keys = torch.cat((old_keys[row], new_k[row]), dim=1).unsqueeze(0)
            values = torch.cat((old_values[row], new_v[row]), dim=1).unsqueeze(0)
            allowed = torch.arange(old_length + query_len, device="cuda").view(1, -1)
            allowed = allowed <= (
                old_length + torch.arange(query_len, device="cuda").view(-1, 1)
            )
            with torch.inference_mode():
                expected = F.scaled_dot_product_attention(
                    q[row : row + 1],
                    keys,
                    values,
                    attn_mask=allowed.view(1, 1, query_len, -1),
                    enable_gqa=True,
                )
            torch.testing.assert_close(
                actual[row : row + 1], expected, rtol=3e-2, atol=3e-2
            )


if __name__ == "__main__":
    unittest.main()
