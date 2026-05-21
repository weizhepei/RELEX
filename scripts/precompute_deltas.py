"""
Precompute and save element-wise deltas from checkpoints.

This script loads all checkpoints, computes absolute deltas (θ_t - θ_0),
and saves them in a memory-mapped format for efficient loading during training.

For 1.5B params × 145 timesteps × 4 bytes = ~870GB, we use chunked storage:
- One file per parameter tensor (338 files)
- Each file: [n_timesteps, *param_shape] as memory-mapped array

Supports incremental computation: if deltas already exist for steps 1-145,
running again with steps 1-500 will only compute steps 146-500 (copying
existing data to resized arrays).

Usage:
  python precompute_deltas.py \
      --base_model Qwen/Qwen2.5-Math-1.5B \
      --checkpoint_dir <path>/RLVR-Qwen2.5-Math-1.5B \
      --output_dir outputs/deltas/Qwen2.5-Math-1.5B \
      --max_checkpoints 500 \
      --num_workers 8

Deltas are stored as fp16 (half precision) for compact disk usage.
Use --force to recompute from scratch ignoring any existing partial deltas.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def extract_step_number(checkpoint_path: str) -> Optional[int]:
    """Extract step number from checkpoint path."""
    match = re.search(r'global_step[_-]?(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def find_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoint directories."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    for item in checkpoint_dir.iterdir():
        if item.is_dir() and re.match(r'global_step[_-]?\d+', item.name):
            step_num = extract_step_number(item.name)
            if step_num is not None:
                possible_paths = [
                    item / "actor" / "huggingface",
                    item / "huggingface",
                    item,
                ]
                for path in possible_paths:
                    if path.exists() and (path / "config.json").exists():
                        checkpoints.append((step_num, str(path)))
                        break

    return sorted(checkpoints, key=lambda x: x[0])


def load_state_dict_fast(
    model_path: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Load state dict directly from safetensors (fast path)."""
    model_path = Path(model_path)

    # Find safetensors file(s)
    safetensor_files = list(model_path.glob("*.safetensors"))

    if not safetensor_files:
        # Fallback to slow path if no safetensors
        return load_state_dict_slow(model_path, dtype)

    state_dict = {}
    for sf_file in safetensor_files:
        # Load directly - much faster than AutoModelForCausalLM
        tensors = load_safetensors(str(sf_file))
        for k, v in tensors.items():
            state_dict[k] = v.to(dtype)

    return state_dict


def load_state_dict_slow(
    model_path: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Load state dict via HuggingFace (slow fallback)."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return state_dict


def load_state_dict(
    model_path: str,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Load model state dict from checkpoint (auto-selects fast/slow path)."""
    model_path = Path(model_path)

    # If it's a HuggingFace model name (not a path), use slow path
    if not model_path.exists():
        return load_state_dict_slow(str(model_path), dtype)

    # Otherwise use fast safetensors loading
    return load_state_dict_fast(model_path, dtype)


def extract_module_type(param_name: str) -> str:
    """Extract module type from parameter name."""
    patterns = [
        r'self_attn\.q_proj',
        r'self_attn\.k_proj',
        r'self_attn\.v_proj',
        r'self_attn\.o_proj',
        r'mlp\.gate_proj',
        r'mlp\.up_proj',
        r'mlp\.down_proj',
        r'input_layernorm',
        r'post_attention_layernorm',
        r'embed_tokens',
        r'norm',
        r'lm_head',
    ]

    for pattern in patterns:
        if re.search(pattern, param_name):
            return pattern.replace(r'\.', '.').replace('\\', '')

    return 'other'


def extract_layer_idx(param_name: str) -> int:
    """Extract layer index from parameter name."""
    match = re.search(r'layers\.(\d+)\.', param_name)
    if match:
        return int(match.group(1))
    return -1  # Non-layer params


def sanitize_filename(param_name: str) -> str:
    """Convert parameter name to valid filename."""
    return param_name.replace('.', '_').replace('/', '_')


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RELEX: deltas are always stored as fp16 (half precision).
    # The original `--dtype` flag has been removed; fp16 halves disk usage
    # vs fp32 without measurable impact on the predicted-checkpoint accuracy
    # (verified for Qwen2.5-Math-1.5B / Qwen3-4B-Base / Qwen3-8B-Base).
    args.dtype = "float16"
    dtype_torch = torch.float16
    dtype_np = np.float16

    # Find checkpoints
    print(f"Finding checkpoints in {args.checkpoint_dir}...")
    checkpoints = find_checkpoints(args.checkpoint_dir)

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Limit to max_checkpoints if specified
    total_found = len(checkpoints)
    if args.max_checkpoints is not None and args.max_checkpoints < total_found:
        checkpoints = checkpoints[:args.max_checkpoints]
        print(f"Found {total_found} checkpoints, using first {args.max_checkpoints}")
    else:
        print(f"Found {total_found} checkpoints")

    n_timesteps = len(checkpoints)
    steps = [s for s, _ in checkpoints]
    print(f"Steps: {steps[:5]}...{steps[-3:]}")

    # Check for existing deltas (incremental mode)
    metadata_path = output_dir / 'metadata.json'
    existing_metadata = None
    existing_steps = []
    incremental_mode = False

    if metadata_path.exists() and not args.force:
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
        existing_steps = existing_metadata.get('steps', [])
        existing_steps_set = set(existing_steps)

        # Find new steps not already computed
        new_checkpoints = [(s, p) for s, p in checkpoints if s not in existing_steps_set]

        if not new_checkpoints:
            print(f"\nAll {n_timesteps} steps already computed. Nothing to do.")
            print("Use --force to recompute from scratch.")
            return

        if len(new_checkpoints) < len(checkpoints):
            incremental_mode = True
            print(f"\n=== INCREMENTAL MODE ===")
            print(f"Existing deltas: {len(existing_steps)} steps ({existing_steps[:3]}...{existing_steps[-3:]})")
            print(f"New steps to compute: {len(new_checkpoints)} ({[s for s, _ in new_checkpoints[:3]]}...{[s for s, _ in new_checkpoints[-3:]]})")

    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    base_state = load_state_dict(args.base_model, dtype_torch)
    # Handle lm_head based on whether the model ties it to embed_tokens.
    # Tied (Qwen2.5 / Qwen3 / OctoThinker): skip lm_head (redundant with embed_tokens).
    # Untied (GLM-4): include lm_head so its delta is tracked.
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    _is_tied = getattr(_cfg, "tie_word_embeddings", True)
    if _is_tied:
        param_names = [name for name in base_state.keys() if 'lm_head' not in name]
        print(f"tie_word_embeddings=True -> excluding lm_head (delta tracked via embed_tokens)")
    else:
        param_names = list(base_state.keys())
        print(f"tie_word_embeddings=False -> including lm_head as a separate tracked tensor")
    n_params = len(param_names)
    print(f"Found {n_params} parameter tensors")

    # Memory usage warning for parallel mode
    if args.num_workers > 1:
        estimated_memory_gb = (args.num_workers + 1) * 6  # 6GB per checkpoint + base
        print(f"\nParallel mode: {args.num_workers} workers")
        print(f"Estimated peak memory: ~{estimated_memory_gb}GB")
        print(f"  (Reduce --num_workers if you encounter OOM errors)")

    # Create metadata for the full set of steps
    metadata = {
        'base_model': args.base_model,
        'checkpoint_dir': args.checkpoint_dir,
        'n_timesteps': n_timesteps,
        'n_params': n_params,
        'steps': steps,
        'dtype': args.dtype,
        'delta_type': 'absolute',  # θ_t - θ_0
        'params': {}
    }

    for name in param_names:
        shape = list(base_state[name].shape)
        numel = base_state[name].numel()
        metadata['params'][name] = {
            'shape': shape,
            'numel': numel,
            'layer_idx': extract_layer_idx(name),
            'module_type': extract_module_type(name),
            'filename': sanitize_filename(name) + '.npy',
        }

    # Create or resize memory-mapped arrays
    mmap_files = {}

    if incremental_mode:
        # Incremental: resize arrays and copy existing data
        print(f"\nResizing arrays and copying existing data ({len(existing_steps)} → {n_timesteps} timesteps)...")

        # Build step-to-index mapping for existing data
        existing_step_to_idx = {s: i for i, s in enumerate(existing_steps)}

        for name in tqdm(param_names, desc="Resizing mmap files"):
            shape = tuple(base_state[name].shape)
            filename = output_dir / metadata['params'][name]['filename']
            old_filename = str(filename) + '.old'

            # Rename old file
            if filename.exists():
                filename.rename(old_filename)

            # Create new larger array
            full_shape = (n_timesteps,) + shape
            new_mmap = np.memmap(filename, dtype=dtype_np, mode='w+', shape=full_shape)

            # Load old data and copy to new array
            old_shape = (len(existing_steps),) + shape
            old_mmap = np.memmap(old_filename, dtype=dtype_np, mode='r', shape=old_shape)

            # Copy existing data to correct positions in new array
            # Fast path: if existing steps are a contiguous prefix (1,2,3,...,N), bulk copy
            if existing_steps == list(range(1, len(existing_steps) + 1)) and steps[:len(existing_steps)] == existing_steps:
                # Bulk copy - much faster than element-by-element
                new_mmap[:len(existing_steps)] = old_mmap[:]
            else:
                # Slow path: steps are non-contiguous or reordered
                for new_idx, step in enumerate(steps):
                    if step in existing_step_to_idx:
                        old_idx = existing_step_to_idx[step]
                        new_mmap[new_idx] = old_mmap[old_idx]

            # Flush and clean up old mmap
            new_mmap.flush()
            del old_mmap

            # Remove old file
            Path(old_filename).unlink()

            mmap_files[name] = new_mmap

        # Filter checkpoints to only new ones
        checkpoints_to_process = new_checkpoints
        # Build step-to-new-index mapping for writing
        step_to_new_idx = {s: i for i, s in enumerate(steps)}

    else:
        # Fresh computation: create new arrays
        print(f"\nCreating memory-mapped arrays...")

        for name in tqdm(param_names, desc="Creating mmap files"):
            shape = base_state[name].shape
            filename = output_dir / metadata['params'][name]['filename']

            # Create memory-mapped array: [n_timesteps, *param_shape]
            full_shape = (n_timesteps,) + tuple(shape)
            mmap = np.memmap(filename, dtype=dtype_np, mode='w+', shape=full_shape)
            mmap_files[name] = mmap

        checkpoints_to_process = checkpoints
        step_to_new_idx = {s: i for i, s in enumerate(steps)}

    # Save metadata (do this after resizing succeeds)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Worker function for parallel checkpoint processing
    def _process_checkpoint_worker(step: int, path: str) -> Tuple[int, bool]:
        """
        Worker function to load one checkpoint and compute deltas.
        Thread-safe: writes to different mmap slices per checkpoint.

        Returns:
            (step, success): Step number and whether it succeeded
        """
        try:
            # Load checkpoint
            state_dict = load_state_dict(path, dtype_torch)

            # Get the index for this step in the output array
            t = step_to_new_idx[step]

            # Compute and store absolute delta for each parameter
            for name in param_names:
                if name in state_dict:
                    delta = state_dict[name] - base_state[name]
                    # Thread-safe: different steps write to different rows
                    mmap_files[name][t] = delta.numpy().astype(dtype_np)

            # Free memory immediately
            del state_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return step, True

        except Exception as e:
            print(f"Error processing step {step}: {e}")
            return step, False

    # Process checkpoints and compute deltas
    num_workers = args.num_workers
    print(f"\nComputing deltas for {len(checkpoints_to_process)} checkpoints with {num_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all checkpoint processing tasks
        futures = {}
        for step, path in checkpoints_to_process:
            future = executor.submit(_process_checkpoint_worker, step, path)
            futures[future] = step

        # Collect results with progress bar
        failed_steps = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing deltas"):
            step, success = future.result()
            if not success:
                failed_steps.append(step)

        if failed_steps:
            print(f"Warning: {len(failed_steps)} checkpoints failed: {failed_steps}")

    # Flush all mmap files
    print("\nFlushing memory-mapped files...")
    for name, mmap in tqdm(mmap_files.items(), desc="Flushing"):
        mmap.flush()
        del mmap

    # Clean up base state
    del base_state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute total size
    total_size = sum(
        (output_dir / metadata['params'][name]['filename']).stat().st_size
        for name in param_names
    )
    print(f"\nTotal storage: {total_size / 1e9:.2f} GB")
    print(f"Deltas saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute element-wise deltas from checkpoints"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base model name/path"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing per-step Hugging Face checkpoints (one global_step_<N>/ per step)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/deltas",
        help="Output directory for delta files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation from scratch, ignoring existing deltas"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for checkpoint processing (default: 4). "
             "Requires ~6GB memory per worker. Set to 1 for sequential processing."
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to process (default: None = use all). "
             "Uses the first N checkpoints sorted by step number."
    )

    args = parser.parse_args()
    main(args)
