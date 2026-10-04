# Repository Guidelines

## Project Structure & Modules

`hayate/model/` contains Qwen layers, attention, rotary embeddings, and KV cache code. `hayate/engine/` handles request scheduling, sampling, prefix caching, and generation. Shared helpers live in `hayate/utils.py`. The `benchmarks/` package implements the benchmark CLI, with `benchmark.py` as its root entry point; `main.py` is a basic inference example. Tests are in `tests/` and use Python's built-in `unittest` framework.

## Setup, Build, and Run

- `uv sync --locked` installs the pinned dependencies into the project environment. Python 3.12+, CUDA, and an NVIDIA GPU are required for inference.
- `python -m unittest discover -s tests` runs the test suite. CUDA-only attention coverage is skipped when CUDA is unavailable.
- `python main.py` loads `Qwen/Qwen3-4B` and generates a sample response; model files may be downloaded on first use.
- `python benchmark.py --help` lists benchmark options. For example, `python benchmark.py --batch-size 4 --context-tokens 2048 --decode-steps 128` runs a larger workload.
- `uv build` builds the Python package.

## Style and Naming

Follow the existing Python style: four spaces for indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Keep modules focused on their current model, engine, or benchmark responsibilities. Tests use `unittest.TestCase` classes and `test_` method names. No formatter, linter, or coverage threshold is configured, so match nearby code and keep changes easy to review.

## Tests and Hardware

Add or update focused tests under `tests/` for behavior changes. Prefer CPU-safe tests for helpers and cache logic; mark tests that need CUDA with `unittest.skipUnless(torch.cuda.is_available(), ...)`, as the attention tests do. Mention GPU model and CUDA requirements when reporting hardware-specific results.

## Commits and Pull Requests

Recent history uses short, informal commit subjects and has no enforced format. Use a concise subject that says what changed (for example, `Handle empty prefix cache entries`). A pull request should explain the change and its reason, list relevant test or benchmark commands and results, and link a related issue when one exists. Include benchmark settings for performance claims.

## Configuration and Local Files

Keep downloaded model weights, credentials, and local environment files out of commits. The repository ignores `.env`, `.venv/`, `*.safetensors`, and the `Qwen3-4B/` model directory; do not add secrets or large model artifacts to source control.
