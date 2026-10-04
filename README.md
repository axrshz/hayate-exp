# hayate

Learning-focused inference engine for `Qwen/Qwen3-4B` on NVIDIA GPUs.

Features:

- BF16 inference with Flash Attention
- KV caching and continuous batching
- Chunked prefill and greedy decoding
- Variable-length prompt batches
- Supports `torch.compile` and prefix caching

## Setup

Requires Python 3.12+, CUDA, and an NVIDIA GPU.

```bash
uv sync --locked
source .venv/bin/activate
```

## Inference

```python
from hayate.engine.engine import Engine

engine = Engine("Qwen/Qwen3-4B")
result = engine.generate_text("Explain artificial general intelligence")
print(result.response)
```

Sampling is greedy by default (`temperature=0`). Set a positive temperature to
sample from the token distribution, and optionally restrict it with top-k and
top-p filtering:

```python
result = engine.generate_text(
    "Explain artificial general intelligence",
    max_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
```

These options also apply to every prompt in a batched `generate_text` call.

Enable `torch.compile` with default mode:

```python
engine = Engine("Qwen/Qwen3-4B", compile=True)
```

Enable prefix caching:

```python
engine = Engine(
    "Qwen/Qwen3-4B",
    enable_prefix_cache=True,
    prefix_cache_max_tokens=4096,
)
```

Long prompts are processed in chunks of 512 tokens by default. This lets active
requests decode between prompt chunks. Set `prefill_chunk_size` when creating
the engine to tune the number of prompt tokens processed per request per tick:

```python
engine = Engine("Qwen/Qwen3-4B", prefill_chunk_size=256)
```

The engine stores request KV states in a shared paged pool. Its default capacity
is 16,384 total token positions across active requests, with 16-token pages. Set
`max_cache_tokens` and `kv_page_size` when creating the engine to tune this pool.
The configured capacity includes prompt and cached generated-token positions;
requests that exceed the available pool raise `MemoryError`.

## Benchmark

```bash
# Run one request with 512 prompt tokens and 64 output tokens.
python benchmark.py

# Set a larger batch, prompt, and output length.
python benchmark.py --batch-size 4 --context-tokens 2048 --decode-steps 128

# Save the measurements and settings in a JSON file.
python benchmark.py --json benchmark-results.json
```

The benchmark measures prefill latency and throughput, decode step latency and
throughput, and peak VRAM for one fixed workload. Run `python benchmark.py --help`
for controls.
