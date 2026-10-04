import math

import torch

from hayate.engine.request import Request


class Sampler:
    """Sample generated tokens and stop requests when needed."""

    def __init__(self, tokenizer, stop_token_ids: set[int]):
        self.tokenizer = tokenizer
        self.stop_token_ids = stop_token_ids

    @staticmethod
    def validate_parameters(
        temperature: float, top_k: int | None, top_p: float | None
    ) -> None:
        """Reject invalid temperature, top-k, or nucleus sampling settings."""
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite number greater than or equal to 0")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError("top_p must be greater than 0 and at most 1, or None")

    @staticmethod
    def _sample_row(
        logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
    ) -> torch.Tensor:
        """Sample one token after applying temperature and optional filters."""
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scores = logits.float() / temperature
        if top_k is not None and top_k < scores.numel():
            threshold = torch.topk(scores, top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, -torch.inf)

        if top_p is not None and top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            remove_sorted = sorted_probs.cumsum(dim=-1) > top_p
            # Keep the token that crosses the threshold so the remaining set is nonempty.
            remove_sorted[1:] = remove_sorted[:-1].clone()
            remove_sorted[0] = False
            remove = torch.zeros_like(remove_sorted).scatter(
                0, sorted_indices, remove_sorted
            )
            scores = scores.masked_fill(remove, -torch.inf)

        probabilities = torch.softmax(scores, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    def sample(
        self, logits: torch.Tensor, requests: list[Request] | None = None
    ) -> torch.Tensor:
        """Return one token ID per row, using that request's sampling controls."""
        if requests is None:
            # Keep the no-argument path compatible with callers that expect greedy decoding.
            return torch.argmax(logits, dim=-1, keepdim=True)
        if logits.ndim != 2 or logits.shape[0] != len(requests):
            raise ValueError("sampling logits must have one vocabulary row per request")

        sampled = []
        for row, request in zip(logits, requests):
            self.validate_parameters(request.temperature, request.top_k, request.top_p)
            sampled.append(
                self._sample_row(
                    row, request.temperature, request.top_k, request.top_p
                )
            )
        return torch.stack(sampled, dim=0)

    def finalize(self, request: Request, tok: int) -> None:
        """Save one token and decode the response when the request ends."""
        request.tokens.append(tok)
        # A stop token or the length limit ends generation for this request.
        if tok in self.stop_token_ids or len(request.tokens) >= request.max_tokens:
            request.is_completed = True
            request.response = self.tokenizer.decode(request.tokens, skip_special_tokens=True)
