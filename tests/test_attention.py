import unittest

import torch

from hayate.model.attention import GroupedQueryAttention
from hayate.model.rope import compute_rope_params


@unittest.skipUnless(torch.cuda.is_available(), "CUDA varlen attention required")
class VariableLengthAttentionTests(unittest.TestCase):
    """Check packed variable-length attention against separate request runs."""

    def test_packed_batch_matches_individual_flash_attention(self):
        # Use unequal prompt lengths so the packed path must ignore left padding.
        torch.manual_seed(7)
        dtype = torch.bfloat16
        module = GroupedQueryAttention(64, 4, 2, 16, dtype=dtype).cuda().eval()
        lengths = torch.tensor([3, 5], device="cuda", dtype=torch.long)
        pads = 5 - lengths
        x = torch.randn(2, 5, 64, device="cuda", dtype=dtype)
        positions = (torch.arange(5, device="cuda").view(1, -1) - pads.view(-1, 1)).clamp(min=0)
        valid = torch.arange(5, device="cuda").view(1, -1) >= pads.view(-1, 1)
        cu = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
        metadata = (valid, valid, cu, cu, 5, 5, True)
        cos, sin = compute_rope_params(16, context_length=16, device="cuda")

        with torch.inference_mode():
            actual, _, _ = module(x, cos, sin, positions, None, None, metadata)
            # Compare each request's real output positions with an unpadded run.
            for index, length in enumerate(lengths.tolist()):
                start = 5 - length
                expected, _, _ = module(
                    x[index : index + 1, start:],
                    cos,
                    sin,
                    torch.arange(length, device="cuda").view(1, -1),
                    None,
                    None,
                )
                torch.testing.assert_close(
                    actual[index, start:], expected[0], rtol=2e-2, atol=2e-2
                )


if __name__ == "__main__":
    unittest.main()
