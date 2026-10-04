import json
import os
import re
from glob import glob

import torch
from safetensors import SafetensorError, safe_open
from huggingface_hub import snapshot_download


def _load_weight_index(local_dir: str) -> dict | None:
    """Read the checkpoint index when the model uses sharded weight files."""
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _expected_shards(weight_index: dict | None) -> set[str]:
    """Return every shard name listed in a checkpoint index."""
    if weight_index is None:
        return set()
    return set(weight_index.get("weight_map", {}).values())


def _expected_shards_from_filenames(local_dir: str) -> set[str]:
    """Infer missing shard names when no checkpoint index exists."""
    expected = set()
    # The shard name contains its number and the total number of files.
    shard_pattern = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d+)-of-(?P<count>\d+)\.safetensors$")
    for path in glob(os.path.join(local_dir, "*.safetensors")):
        filename = os.path.basename(path)
        match = shard_pattern.match(filename)
        if match is None:
            continue

        prefix = match.group("prefix")
        width = len(match.group("idx"))
        count = int(match.group("count"))
        expected.update(f"{prefix}-{idx:0{width}d}-of-{count:0{width}d}.safetensors" for idx in range(1, count + 1))
    return expected


def _missing_shards(local_dir: str, shards: set[str]) -> list[str]:
    """Return expected shard files that do not exist in the local directory."""
    return sorted(shard for shard in shards if not os.path.isfile(os.path.join(local_dir, shard)))


def _has_usable_weights(local_dir: str) -> bool:
    """Check that the directory contains all weight files expected for its layout."""
    safetensor_files = glob(os.path.join(local_dir, "*.safetensors"))
    if not safetensor_files:
        return False

    weight_index = _load_weight_index(local_dir)
    # An index gives exact shard names; filenames provide a fallback check.
    expected = _expected_shards(weight_index) or _expected_shards_from_filenames(local_dir)
    missing = _missing_shards(local_dir, expected)
    return not missing


def download_weights(repo_id: str, local_dir: str):
    """Return a local checkpoint path, and download files when the checkpoint is incomplete."""
    # A directory argument means the caller already supplied a local checkpoint.
    if os.path.isdir(repo_id):
        return repo_id

    # Reuse complete downloads so startup does not fetch the model again.
    if os.path.isdir(local_dir) and _has_usable_weights(local_dir):
        return local_dir

    print(f"Downloading weights from {repo_id} to {local_dir}/ ...")
    # Model weights, config files, and tokenizer files are enough for inference.
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )
    if not _has_usable_weights(local_dir):
        weight_index = _load_weight_index(local_dir)
        expected = _expected_shards(weight_index) or _expected_shards_from_filenames(local_dir)
        missing = _missing_shards(local_dir, expected)
        if missing:
            raise FileNotFoundError(
                "Downloaded checkpoint is incomplete. Missing shard(s): "
                f"{', '.join(missing)}. Delete '{local_dir}' and rerun, or run "
                "`huggingface-cli download Qwen/Qwen3-4B --local-dir Qwen3-4B`."
            )
        raise FileNotFoundError(f"No safetensor checkpoint files found in '{local_dir}'.")
    return local_dir


def assign(left, right, tensor_name="unknown"):
    """Copy checkpoint values into a model parameter after a shape check."""
    # A mismatch often means that the code and the checkpoint use different models.
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in tensor '{tensor_name}'. "
            f"Left: {tuple(left.shape)}, Right: {tuple(right.shape)}"
        )

    # Weight loading must not create gradient history.
    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right.to(dtype=left.dtype, device=left.device))
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left


def qwen_weight_map(model):
    """Map Hugging Face tensor names to the matching Qwen model parameters."""
    # The checkpoint uses these top-level names for embeddings and final layers.
    weight_map = {
        "model.embed_tokens.weight": model.embed_tokens.weight,
        "model.norm.weight": model.norm.weight,
        "lm_head.weight": model.out_head.weight,
    }

    # Repeat the name mapping for attention, normalization, and feed-forward weights.
    for layer_idx in range(model.num_layers):
        block = model.layers[layer_idx]
        attn = block.self_attn

        weight_map.update(
            {
                f"model.layers.{layer_idx}.self_attn.q_proj.weight": attn.q_proj.weight,
                f"model.layers.{layer_idx}.self_attn.k_proj.weight": attn.k_proj.weight,
                f"model.layers.{layer_idx}.self_attn.v_proj.weight": attn.v_proj.weight,
                f"model.layers.{layer_idx}.self_attn.o_proj.weight": attn.o_proj.weight,
                f"model.layers.{layer_idx}.self_attn.q_norm.weight": attn.q_norm.weight,
                f"model.layers.{layer_idx}.self_attn.k_norm.weight": attn.k_norm.weight,
                f"model.layers.{layer_idx}.input_layernorm.weight": block.input_layernorm.weight,
                f"model.layers.{layer_idx}.mlp.gate_proj.weight": block.mlp.gate_proj.weight,
                f"model.layers.{layer_idx}.mlp.up_proj.weight": block.mlp.up_proj.weight,
                f"model.layers.{layer_idx}.mlp.down_proj.weight": block.mlp.down_proj.weight,
                f"model.layers.{layer_idx}.post_attention_layernorm.weight": block.post_attention_layernorm.weight,
            }
        )

    return weight_map


def load_weights(model, repo_id: str):
    """Download and load all required model tensors from safetensors files."""
    # Hugging Face IDs use their final path component for the local folder name.
    local_dir = repo_id.split("/")[-1]
    local_dir = download_weights(repo_id, local_dir)

    weight_map = qwen_weight_map(model)
    # Some checkpoints share the output head with the token embedding table.
    required_weights = {name for name in weight_map if name != "lm_head.weight"}
    loaded_weights = set()
    lm_head_loaded = False
    weight_index = _load_weight_index(local_dir)

    if weight_index is not None:
        # Check index entries and files before opening checkpoint shards.
        checkpoint_weights = set(weight_index.get("weight_map", {}))
        missing_from_index = sorted(required_weights - checkpoint_weights)
        if missing_from_index:
            missing_preview = ", ".join(missing_from_index[:5])
            if len(missing_from_index) > 5:
                missing_preview += ", ..."
            raise KeyError(f"Checkpoint index is missing required tensors: {missing_preview}")

        missing_shards = _missing_shards(local_dir, _expected_shards(weight_index))
        if missing_shards:
            raise FileNotFoundError(
                "Checkpoint is incomplete. Missing shard(s): "
                f"{', '.join(missing_shards)}. Delete '{local_dir}' and rerun the benchmark "
                "so the Hugging Face download can resume cleanly."
            )
    else:
        missing_shards = _missing_shards(local_dir, _expected_shards_from_filenames(local_dir))
        if missing_shards:
            raise FileNotFoundError(
                "Checkpoint appears to be an incomplete sharded safetensors download. "
                f"Missing shard(s): {', '.join(missing_shards)}."
            )

    # Read shards in a stable order and ignore tensors that the model does not use.
    for file in sorted(glob(os.path.join(local_dir, "*.safetensors"))):
        try:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    if weight_name not in weight_map:
                        continue

                    # Copy one tensor at a time so the full checkpoint stays off memory.
                    assign(weight_map[weight_name], f.get_tensor(weight_name), weight_name)
                    loaded_weights.add(weight_name)

                    if weight_name == "lm_head.weight":
                        lm_head_loaded = True
        except SafetensorError as exc:
            raise RuntimeError(
                f"Failed to read checkpoint shard '{file}'. The file may be incomplete or "
                f"corrupt; delete '{local_dir}' and rerun to redownload it."
            ) from exc

    # A valid shard set can still omit a tensor required by this model.
    missing_weights = sorted(required_weights - loaded_weights)
    if missing_weights:
        missing_preview = ", ".join(missing_weights[:5])
        if len(missing_weights) > 5:
            missing_preview += ", ..."
        raise KeyError(f"Missing required checkpoint tensors: {missing_preview}")

    # Qwen checkpoints can tie output logits to the input embedding weights.
    if not lm_head_loaded:
        model.out_head.weight = model.embed_tokens.weight

    return local_dir
