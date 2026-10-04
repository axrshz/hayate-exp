# Hayate places model tensors on this device after it checks CUDA support.
device = "cuda"

# These names are valid values for the optional torch.compile setting.
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")

# The scheduler admits at most this many active requests in one batch.
MAX_BATCH_SIZE = 20

# Hayate splits larger prompt groups to limit temporary prefill memory.
MAX_PREFILL_BATCH = 8

# Each request processes at most this many prompt tokens per scheduler tick.
DEFAULT_PREFILL_CHUNK_SIZE = 512

# Total logical token positions held by the default shared paged KV pool.
DEFAULT_KV_CACHE_TOKENS = 16_384

# KV page size also sets the key block size used by FlexAttention.
DEFAULT_KV_PAGE_SIZE = 16

# Prefix caching can store this many token positions by default.
DEFAULT_PREFIX_CACHE_MAX_TOKENS = 4096
