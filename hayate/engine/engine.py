from typing import List, Union

import torch
from transformers import AutoTokenizer, GenerationConfig

from hayate.engine.constants import (
    COMPILE_MODES,
    DEFAULT_KV_CACHE_TOKENS,
    DEFAULT_KV_PAGE_SIZE,
    DEFAULT_PREFILL_CHUNK_SIZE,
    DEFAULT_PREFIX_CACHE_MAX_TOKENS,
    MAX_BATCH_SIZE,
    MAX_PREFILL_BATCH,
    device,
)
from hayate.engine.prefix_cache import PrefixCache
from hayate.engine.paged_cache import PagedKVCacheManager
from hayate.engine.request import Request
from hayate.engine.sampler import Sampler
from hayate.engine.scheduler import Scheduler
from hayate.model import Qwen3Model
from hayate.model.cache import Cache
from hayate.utils import load_weights


class Engine:
    """Load Qwen weights and coordinate prompt prefill, token decode, and caching."""

    def __init__(
        self,
        model_name: str,
        compile: bool = False,
        compile_mode: str = "default",
        enable_prefix_cache: bool = False,
        prefix_cache_max_tokens: int = DEFAULT_PREFIX_CACHE_MAX_TOKENS,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        max_cache_tokens: int = DEFAULT_KV_CACHE_TOKENS,
        kv_page_size: int = DEFAULT_KV_PAGE_SIZE,
    ):
        """Create an inference engine for a local or Hugging Face Qwen checkpoint."""
        # Hayate uses CUDA tensors and reports a clear error before model setup.
        if not torch.cuda.is_available():
            raise RuntimeError("hayate requires an NVIDIA CUDA GPU")
        # Reject invalid compile settings before the engine loads a large model.
        if compile_mode not in COMPILE_MODES:
            raise ValueError(
                f"unknown compile mode '{compile_mode}'. Valid options: {COMPILE_MODES}"
            )
        if (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size < 1
        ):
            raise ValueError("prefill_chunk_size must be a positive integer")
        if (
            isinstance(max_cache_tokens, bool)
            or not isinstance(max_cache_tokens, int)
            or max_cache_tokens < 1
        ):
            raise ValueError("max_cache_tokens must be a positive integer")
        if (
            isinstance(kv_page_size, bool)
            or not isinstance(kv_page_size, int)
            or kv_page_size < 1
            or kv_page_size & (kv_page_size - 1)
        ):
            raise ValueError("kv_page_size must be a positive power of two")
        self.prefill_chunk_size = prefill_chunk_size
        # Create every model layer on CUDA before copying checkpoint weights into it.
        with torch.device("cuda"):
            self.model = Qwen3Model()
        # Keep architecture sizes for cache shapes and context-window checks.
        self.num_layers = self.model.num_layers
        self.num_kv_groups = self.model.num_kv_groups
        self.head_dim = self.model.head_dim
        self.max_position_embeddings = self.model.max_position_embeddings
        model_dtype = self.model.embed_tokens.weight.dtype
        # Empty storage avoids keeping temporary random parameter values during weight loading.
        self.model.to_empty(device=device)
        self.model.initialize_rope(device)
        # Weight loading also downloads missing checkpoint files when needed.
        model_dir = load_weights(self.model, model_name)
        self.model.eval()
        # Compilation is optional because setup can take time and use extra memory.
        if compile:
            if compile_mode == "default":
                self.model = torch.compile(self.model, dynamic=True)
            else:
                self.model = torch.compile(self.model, dynamic=True, mode=compile_mode)
        # Load tokenizer files from the same directory as the model weights.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True, local_files_only=True
        )
        self.stop_token_ids = self._load_stop_token_ids(model_dir)
        self.scheduler = Scheduler()
        self.prefix_cache = (
            PrefixCache(prefix_cache_max_tokens) if enable_prefix_cache else None
        )
        self.sampler = Sampler(self.tokenizer, self.stop_token_ids)
        # Reserve one shared pool so requests can use non-contiguous KV pages.
        self.paged_cache = PagedKVCacheManager(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_groups,
            head_dim=self.head_dim,
            max_cache_tokens=max_cache_tokens,
            page_size=kv_page_size,
            max_requests=MAX_BATCH_SIZE,
            dtype=model_dtype,
            device=device,
        )

    def _load_stop_token_ids(self, model_name: str) -> set[int]:
        """Return configured end tokens, including the tokenizer's default token."""
        token_ids = set()
        # Some checkpoints define more than one token that ends generation.
        try:
            generation_config = GenerationConfig.from_pretrained(model_name)
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, int):
                token_ids.add(eos_token_id)
            elif eos_token_id is not None:
                token_ids.update(int(tok) for tok in eos_token_id)
        except Exception:
            # The tokenizer value below remains available if config loading fails.
            pass

        if self.tokenizer.eos_token_id is not None:
            token_ids.add(int(self.tokenizer.eos_token_id))
        return token_ids

    def add_request(self, request: Request):
        """Prepare a request and place it in the scheduler's waiting queue."""
        self._prepare_request(request)
        self.scheduler.add(request)

    def _prepare_request(self, request: Request):
        """Tokenize the prompt, check its size, and reuse a matching cached prefix."""
        self.sampler.validate_parameters(
            request.temperature, request.top_k, request.top_p
        )
        # Store token IDs once so the engine does not encode the same prompt again.
        if not request.prompt_tokens:
            request.prompt_tokens = self.tokenizer.encode(request.prompt)

        self._validate_request_lengths(request)

        # Keep an existing cache when a caller resumes a prepared request.
        if request.kv_cache is not None and request.kv_cache.length > 0:
            return

        # Every request owns its active cache, even when no prefix can be reused.
        request.kv_cache = Cache()
        if self.prefix_cache is None:
            return

        # Prefill must process one prompt token to calculate the first output token.
        max_prefix_len = max(len(request.prompt_tokens) - 1, 0)
        match = self.prefix_cache.get(
            request.prompt_tokens, max_prefix_len=max_prefix_len
        )
        if match.cache is None or match.length == 0:
            return

        request.kv_cache = match.cache

    def _validate_request_lengths(self, request: Request):
        """Check token limits before any request enters the model."""
        if request.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not request.prompt_tokens:
            raise ValueError("prompt must tokenize to at least one token")

        # The last sampled token ends generation and does not enter the KV cache.
        required_positions = len(request.prompt_tokens) + request.max_tokens - 1
        if required_positions > self.max_position_embeddings:
            raise ValueError(
                "request exceeds model context window: "
                f"prompt tokens ({len(request.prompt_tokens)}) + generated-token positions "
                f"({request.max_tokens - 1}) = {required_positions}, "
                f"max supported positions = {self.max_position_embeddings}"
            )

    def clear_prefix_cache(self):
        """Remove every reusable prompt prefix from the optional prefix cache."""
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    def _store_prompt_prefix(self, request: Request):
        """Save the processed prompt states for later requests with the same prefix."""
        if self.prefix_cache is None or request.paged_cache is None:
            return
        if request.paged_cache.length < len(request.prompt_tokens):
            return
        snapshot = self.paged_cache.snapshot(
            request.paged_cache, len(request.prompt_tokens)
        )
        self.prefix_cache.put(request.prompt_tokens, snapshot)
        snapshot.reset()

    def release_request_cache(self, request: Request) -> None:
        """Release a request's pages and any temporary contiguous cache."""
        if request.paged_cache is not None:
            self.paged_cache.release(request.paged_cache)
            request.paged_cache = None
        if request.kv_cache is not None:
            request.kv_cache.reset()
            request.kv_cache = None

    def _forward_pass(self, tokens, requests: List[Request], pad_lengths_py=None):
        """Run one model step and append its new key and value states to each request."""
        if pad_lengths_py is not None and any(pad_lengths_py):
            raise ValueError("paged attention requires unpadded request chunks")
        T = tokens.shape[1]
        handles = [request.paged_cache for request in requests]
        if any(handle is None for handle in handles):
            raise RuntimeError("every request must own paged KV storage before inference")
        handles = [handle for handle in handles if handle is not None]
        paged_batch = self.paged_cache.begin_batch(handles, T)

        # Generation does not need gradients, so inference mode reduces overhead.
        with torch.inference_mode():
            logits, _, _ = self.model(tokens, paged_cache=paged_batch)
        self.paged_cache.commit_batch(handles, T)
        return logits[:, -1, :]

    def prefill_batch(self, requests: List[Request]):
        """Process one prompt chunk per request and sample completed prompts."""
        chunks_by_length: dict[int, list[tuple[Request, list[int], bool]]] = {}
        for request in requests:
            if not request.prompt_tokens:
                request.prompt_tokens = self.tokenizer.encode(request.prompt)
            if not request.prompt_tokens:
                raise ValueError("prompt must tokenize to at least one token")

            # The final prompt token must pass through the model to predict output token one.
            max_prefix_len = len(request.prompt_tokens) - 1
            if request.paged_cache is None:
                request.paged_cache = self.paged_cache.allocate_for_request(
                    request.kv_cache
                )
                if request.kv_cache is not None:
                    request.kv_cache.reset()
                    request.kv_cache = None
            cache_pos = request.paged_cache.length
            if cache_pos > max_prefix_len:
                self.paged_cache.truncate(request.paged_cache, max_prefix_len)
                cache_pos = max_prefix_len

            chunk_end = min(
                cache_pos + self.prefill_chunk_size, len(request.prompt_tokens)
            )
            chunk = request.prompt_tokens[cache_pos:chunk_end]
            chunks_by_length.setdefault(len(chunk), []).append(
                (request, chunk, chunk_end == len(request.prompt_tokens))
            )

        # Equal-length groups avoid padded writes into physical pages.
        for _, items in chunks_by_length.items():
            chunk_requests = [item[0] for item in items]
            tokens = torch.tensor([item[1] for item in items], device=device)
            last_logits = self._forward_pass(tokens, chunk_requests)
            completed = [item for item in items if item[2]]
            if not completed:
                continue

            completed_indices = [
                index for index, item in enumerate(items) if item[2]
            ]
            completed_requests = [item[0] for item in completed]
            next_tokens = self.sampler.sample(
                last_logits[completed_indices], completed_requests
            )
            for request, token in zip(completed_requests, next_tokens.flatten().tolist()):
                self.sampler.finalize(request, token)
                request.is_prefill = False
                self._store_prompt_prefix(request)

    def decode_batch(self, requests: List[Request]):
        """Use each request's latest token to predict one more token."""
        # Decode processes one prior output token for each unfinished request.
        tokens = torch.tensor([[r.tokens[-1]] for r in requests], device=device)

        last_logits = self._forward_pass(tokens, requests, pad_lengths_py=None)
        next_tokens = self.sampler.sample(last_logits, requests)
        next_token_ids = next_tokens.flatten().tolist()

        for i, request in enumerate(requests):
            tok = next_token_ids[i]
            self.sampler.finalize(request, tok)

    def generate(self):
        """Run one scheduler tick and return whether any request remains active."""
        previous_batch = list(self.scheduler.current_batch)
        batch = self.scheduler.tick()
        active_ids = {id(request) for request in batch}
        for request in previous_batch:
            if request.is_completed and id(request) not in active_ids:
                self.release_request_cache(request)
        if not batch:
            return False

        # New requests need prompt prefill; active requests need one-token decode.
        prefill_requests = [r for r in batch if r.is_prefill]
        decode_requests = [r for r in batch if not r.is_prefill]

        # Smaller prefill groups limit memory use for long prompts.
        for i in range(0, len(prefill_requests), MAX_PREFILL_BATCH):
            chunk = prefill_requests[i : i + MAX_PREFILL_BATCH]
            self.prefill_batch(chunk)

        if decode_requests:
            self.decode_batch(decode_requests)

        return True

    def generate_text(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ):
        """Generate text with greedy or temperature, top-k, and top-p sampling."""
        # Normalize one prompt to the same list path used for batch requests.
        if isinstance(prompts, str):
            prompts = [prompts]

        requests = []
        # Each request receives an ID that stays unique for this engine.
        for prompt in prompts:
            req = Request(
                id=self.scheduler.request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            self.scheduler.request_id += 1
            self.add_request(req)
            requests.append(req)

        # Each tick handles one prompt step or one decode step for active requests.
        while self.generate():
            pass

        return requests[0] if len(requests) == 1 else requests
