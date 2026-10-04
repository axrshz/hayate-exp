from dataclasses import dataclass, field
from typing import List

from hayate.engine.paged_cache import PagedCacheHandle
from hayate.model.cache import Cache


@dataclass
class Request:
    """Store one prompt, its generated tokens, and its progress through inference."""

    # The scheduler assigns an ID and uses these fields to track request limits.
    id: int = 0
    prompt: str = ""
    max_tokens: int = 100

    # Token IDs avoid repeated text encoding during model execution.
    prompt_tokens: List[int] = field(default_factory=list)
    tokens: List[int] = field(default_factory=list)
    kv_cache: Cache | None = None
    paged_cache: PagedCacheHandle | None = None

    # The engine uses these flags to choose prefill, decode, and cleanup steps.
    is_completed: bool = False
    is_prefill: bool = True

    # The sampler fills this field after a stop condition completes the request.
    response: str | None = None

    # Sampling controls are stored per request so mixed batches are supported.
    # Temperature zero preserves the engine's original greedy decoding behavior.
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
