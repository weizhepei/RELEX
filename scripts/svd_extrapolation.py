"""
SVD-based low-rank extrapolation of model weight deltas.

Motivation:
  Sparse linear extrapolation selects "active" parameters via a magnitude threshold
  (epsilon) and fits a linear trend per parameter. This works but has drawbacks:
  the threshold is a heuristic, and the standard basis (individual params) is not
  necessarily the best coordinate system for capturing weight dynamics.

  SVD-based extrapolation addresses both issues. Instead of thresholding individual
  parameters, it finds the optimal low-rank subspace of weight changes via SVD.
  The rank r replaces epsilon as a cleaner hyperparameter — it directly controls
  the dimensionality of the extrapolation subspace, analogous to LoRA rank.

Method:
  For each weight tensor with T history checkpoints and d parameters:
  1. Stack flattened deltas (theta_t - theta_0) into matrix M [T, d]
  2. Compute truncated SVD: M ≈ U_r S_r V_r^T
     - V_r [d, r]: the r principal directions of weight change
     - U_r S_r [T, r]: temporal coefficients in this subspace
     Since T << d, we use the Gram matrix trick: G = M M^T [T,T] is tiny
  3. Project deltas into the rank-r subspace for reconstruction or extrapolation

Two modes:
  reconstruct: Project actual deltas into rank-r space and back. Tests whether the
               weight dynamics are truly low-rank — if performance is preserved,
               the rank-r subspace captures the essential training signal.
  extrapolate: Compute temporal coefficients C = M @ V_r [T, r], fit a linear trend
               on each of the r components, and predict future steps. Optionally
               uses a sliding window (--window) to fit on only the most recent
               W timesteps, capturing local trends.

Explained variance:
  For each tensor, we report the fraction of total variance captured by rank r:
    explained_var = (σ₁² + σ₂² + ... + σᵣ²) / (σ₁² + σ₂² + ... + σ_T²)
  where σᵢ are the singular values (sqrt of eigenvalues of G = M M^T).
  This measures how much of the "energy" in weight changes is captured by the
  rank-r subspace. It is a property of the data matrix M — computed purely from
  SVD, without any reconstruction or evaluation.
  - 1.0: perfect, all variance captured (r = full rank)
  - 0.96: 96% captured, 4% discarded — typically sufficient
  - 0.0: no variance (e.g., layernorm weights that don't change during training)
  High explained variance is necessary but not sufficient for preserving
  performance — downstream eval of reconstructed checkpoints is needed to confirm.

Hyperparameters:
  r (--rank): SVD rank — number of principal directions to keep. Start with
              reconstruct mode to find the smallest r that preserves performance.
  W (--window): Sliding window for linear fit (default: -1 = full history).
              Smaller W captures more recent/local trends.

Usage examples:

# (a) Reconstruction within the observed window (oracle, no extrapolation):
python svd_extrapolation.py \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --base_model Qwen/Qwen2.5-Math-1.5B \
    --output_dir outputs/reconstruct/Qwen2.5-Math-1.5B/rank1 \
    --history_steps 500 --rank 1 \
    --target_steps 100,200,300,400,500 \
    --mode reconstruct

# (b) Extrapolation: paper-default RELEX (rank-1, linear fit) from T_cut=75:
python svd_extrapolation.py \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --base_model Qwen/Qwen2.5-Math-1.5B \
    --output_dir outputs/relex/Qwen2.5-Math-1.5B/cutoff75 \
    --svd_cache_dir outputs/svd_cache/Qwen2.5-Math-1.5B/cutoff75 \
    --history_steps 75 --rank 1 --fit_method linear \
    --target_steps 100,200,300,400,500 \
    --mode extrapolate

# (c) Higher-rank or alternative fits (ablation):
python svd_extrapolation.py \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --base_model Qwen/Qwen2.5-Math-1.5B \
    --output_dir outputs/relex_ablation/Qwen2.5-Math-1.5B/cutoff75_rank5 \
    --history_steps 75 --rank 5 --fit_method linear \
    --target_steps 100,200,300,400,500 \
    --mode extrapolate
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

# Numerical convention
# ────────────────────
# Deltas (θ_t − θ_0) are stored on disk as **float16** (half precision) to
# halve disk usage with no measurable impact on the predicted-checkpoint
# accuracy; see `precompute_deltas.py` and metadata.json `dtype` field.
#
# In-RAM compute arrays (Gram matrix G, chunked M, V_r, etc.) are float32:
#   - `np.linalg.eigh` does not support fp16 (LAPACK limitation).
#   - fp16 matmul on large embedding tensors can overflow during
#     accumulation; fp32 is the standard safe default.
#   - fp64 would 2× the chunk RAM with no measurable accuracy gain.
# fp16 deltas read from disk are implicitly upcast to fp32 on assignment
# into the compute arrays.

def load_delta_row(filepath, shape, idx, dtype=np.float16):
    """Load a single timestep's delta from a .npy file using offset.

    `filepath` may be a single Path (single delta_dir) or a list of
    `(Path, n_rows)` tuples (compound delta_dirs, e.g. phase1+phase2).
    For the compound form, idx is routed to the segment containing it.
    """
    if isinstance(filepath, list):
        for path_i, n_i in filepath:
            if idx < n_i:
                return load_delta_row(path_i, shape, idx, dtype)
            idx -= n_i
        raise IndexError(f"idx {idx} exceeds compound filepath length")
    numel = int(np.prod(shape))
    offset = idx * numel * np.dtype(dtype).itemsize
    return np.fromfile(filepath, dtype=dtype, count=numel, offset=offset)


def load_delta_row_slice(filepath, numel, idx, start, end, dtype=np.float16):
    """Load a slice [start:end] of a single timestep's flattened delta.

    Supports compound filepath (list of (Path, n_rows)); see load_delta_row.
    """
    if isinstance(filepath, list):
        for path_i, n_i in filepath:
            if idx < n_i:
                return load_delta_row_slice(path_i, numel, idx, start, end, dtype)
            idx -= n_i
        raise IndexError(f"idx {idx} exceeds compound filepath length")
    itemsize = np.dtype(dtype).itemsize
    offset = (idx * numel + start) * itemsize
    return np.fromfile(filepath, dtype=dtype, count=end - start, offset=offset)


def compute_svd(M, rank):
    """Compute truncated SVD via Gram matrix (efficient when T << d).

    Args:
        M: array [T, d] — stacked flattened deltas
        rank: number of singular vectors to keep

    Returns:
        V_r: array [d, rank] — right singular vectors
        S: array [rank] — singular values (descending)
        U_r: array [T, rank] — left singular vectors
        explained_var: float — fraction of variance explained by top-r components
    """
    T, d = M.shape
    rank = min(rank, T)  # can't exceed number of timesteps

    # Gram matrix G = M M^T, shape [T, T]
    G = M @ M.T

    # Eigendecompose (returns ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Explained variance
    total_var = np.sum(np.maximum(eigenvalues, 0))
    top_var = np.sum(np.maximum(eigenvalues[:rank], 0))
    explained_var = top_var / total_var if total_var > 0 else 0.0

    # Truncate to rank
    eigenvalues = eigenvalues[:rank]
    U_r = eigenvectors[:, :rank]  # [T, rank]

    # Singular values
    S = np.sqrt(np.maximum(eigenvalues, 0))

    # Right singular vectors: V_r = M^T @ U_r / S
    V_r = M.T @ U_r  # [d, rank]
    for i in range(rank):
        sv_threshold = S[0] * 1e-6 if S[0] > 0 else 1e-12
        if S[i] > sv_threshold:
            V_r[:, i] /= S[i]
        else:
            V_r[:, i] = 0.0

    return V_r, S, U_r, explained_var


def compute_svd_streaming(filepath, numel, history_indices, rank, dtype=np.float16, chunk_d=100_000_000):
    """Compute truncated SVD without materializing full [T, d] matrix.

    Chunks along the feature dimension d. Peak memory: O(T * chunk_d) instead of O(T * d).
    Two passes through the file: one for Gram matrix G, one for right singular vectors V_r.
    """
    T = len(history_indices)
    rank = min(rank, T)

    # Pass 1: Build Gram matrix G = M M^T by chunking along d
    # G = sum_chunks(M_chunk @ M_chunk^T) since M = [M_1 | M_2 | ...]
    G = np.zeros((T, T), dtype=np.float32)
    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        M_chunk = np.zeros((T, d_end - d_start), dtype=np.float32)
        for i, hi in enumerate(history_indices):
            M_chunk[i] = load_delta_row_slice(filepath, numel, hi, d_start, d_end, dtype)
        G += M_chunk @ M_chunk.T
        del M_chunk

    # Eigendecompose (eigh internally uses float64)
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(np.maximum(eigenvalues, 0))
    top_var = np.sum(np.maximum(eigenvalues[:rank], 0))
    explained_var = top_var / total_var if total_var > 0 else 0.0

    eigenvalues = eigenvalues[:rank]
    U_r = eigenvectors[:, :rank].astype(np.float32)  # [T, rank]
    S = np.sqrt(np.maximum(eigenvalues, 0)).astype(np.float32)

    # Pass 2: Build V_r = M^T @ U_r / S by chunking along d
    V_r = np.zeros((numel, rank), dtype=np.float32)
    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        M_chunk = np.zeros((T, d_end - d_start), dtype=np.float32)
        for i, hi in enumerate(history_indices):
            M_chunk[i] = load_delta_row_slice(filepath, numel, hi, d_start, d_end, dtype)
        V_chunk = M_chunk.T @ U_r  # [chunk_d, rank]
        for k in range(rank):
            sv_threshold_k = S[0] * 1e-6 if S[0] > 0 else 1e-12
            if S[k] > sv_threshold_k:
                V_chunk[:, k] /= S[k]
            else:
                V_chunk[:, k] = 0.0
        V_r[d_start:d_end] = V_chunk
        del M_chunk, V_chunk

    return V_r, S, U_r, explained_var


def compute_gram_streaming(filepath, numel, history_indices, rank, dtype=np.float16, chunk_d=100_000_000):
    """Pass 1 only: build Gram matrix G and eigendecompose. Returns U_r, S, explained_var.

    Used when V_r = [numel, rank] is too large to store in RAM. The caller uses U_r and S
    to compute C analytically (C = U_r * S) and to reconstruct via on-the-fly V_r chunks.
    """
    T = len(history_indices)
    rank = min(rank, T)

    G = np.zeros((T, T), dtype=np.float32)
    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        M_chunk = np.zeros((T, d_end - d_start), dtype=np.float32)
        for i, hi in enumerate(history_indices):
            M_chunk[i] = load_delta_row_slice(filepath, numel, hi, d_start, d_end, dtype)
        G += M_chunk @ M_chunk.T
        del M_chunk

    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = np.sum(np.maximum(eigenvalues, 0))
    top_var = np.sum(np.maximum(eigenvalues[:rank], 0))
    explained_var = top_var / total_var if total_var > 0 else 0.0

    eigenvalues = eigenvalues[:rank]
    U_r = eigenvectors[:, :rank].astype(np.float32)
    S = np.sqrt(np.maximum(eigenvalues, 0)).astype(np.float32)

    return U_r, S, explained_var


def _vr_chunk(filepath, numel, history_indices, d_start, d_end, U_r, S, dtype=np.float16):
    """Compute V_r[d_start:d_end, :] on the fly from M_hist chunk."""
    T = len(history_indices)
    M_chunk = np.zeros((T, d_end - d_start), dtype=np.float32)
    for i, hi in enumerate(history_indices):
        M_chunk[i] = load_delta_row_slice(filepath, numel, hi, d_start, d_end, dtype)
    V_chunk = M_chunk.T @ U_r  # [chunk_d, rank]
    _sv_thresh = S[0] * 1e-6 if S[0] > 0 else 1e-12
    S_safe = np.where(S > _sv_thresh, S, 1.0)
    V_chunk /= S_safe[np.newaxis, :]
    V_chunk[:, S <= _sv_thresh] = 0.0
    return V_chunk


def reconstruct_streaming(filepath, numel, history_indices, target_indices, U_r, S, rank,
                           dtype=np.float16, chunk_d=100_000_000):
    """Reconstruct deltas for target steps without materializing full V_r [numel, rank].

    Pass 2: accumulate c_t[j] = delta_t[j] @ V_r chunk-by-chunk.
    Pass 3: compute delta_lr[j, chunk] = c_t[j] @ V_r_chunk.T chunk-by-chunk.

    Returns:
        c_targets: [n_targets, rank]
        delta_lrs: [n_targets, numel]
    """
    T = len(history_indices)
    n_targets = len(target_indices)

    # Pass 2: accumulate c_t for each target step
    c_targets = np.zeros((n_targets, rank), dtype=np.float32)
    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        V_chunk = _vr_chunk(filepath, numel, history_indices, d_start, d_end, U_r, S, dtype)
        for j, ti in enumerate(target_indices):
            delta_t_chunk = load_delta_row_slice(filepath, numel, ti, d_start, d_end, dtype)
            c_targets[j] += delta_t_chunk @ V_chunk
        del V_chunk

    # Pass 3: reconstruct chunk-by-chunk
    delta_lrs = np.zeros((n_targets, numel), dtype=np.float32)
    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        V_chunk = _vr_chunk(filepath, numel, history_indices, d_start, d_end, U_r, S, dtype)
        delta_lrs[:, d_start:d_end] = c_targets @ V_chunk.T
        del V_chunk

    return c_targets, delta_lrs


def predict_streaming(filepath, numel, history_indices, c_preds_array, U_r, S,
                      dtype=np.float16, chunk_d=100_000_000):
    """Compute delta = c_preds @ V_r.T chunk-by-chunk without materializing V_r.

    Args:
        c_preds_array: [n_targets, rank] — predicted coefficients for each target step

    Returns:
        delta_preds: [n_targets, numel]
    """
    n_targets = len(c_preds_array)
    delta_preds = np.zeros((n_targets, numel), dtype=np.float32)

    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        V_chunk = _vr_chunk(filepath, numel, history_indices, d_start, d_end, U_r, S, dtype)
        delta_preds[:, d_start:d_end] = c_preds_array @ V_chunk.T
        del V_chunk

    return delta_preds


def fit_linear(C):
    """Fit linear regression on temporal SVD coefficients.

    Args:
        C: array [T, rank] — temporal coefficients (each column is one component over time)

    Returns:
        slopes: array [rank]
        intercepts: array [rank]
    """
    T, rank = C.shape
    t = np.arange(T, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_var = np.sum(t_centered ** 2)

    slopes = np.zeros(rank, dtype=np.float64)
    intercepts = np.zeros(rank, dtype=np.float64)

    for i in range(rank):
        y = C[:, i].astype(np.float64)
        y_mean = y.mean()
        slopes[i] = np.sum(t_centered * (y - y_mean)) / t_var
        intercepts[i] = y_mean - slopes[i] * t_mean

    return slopes, intercepts


def _sin_func(t, A, f, phi, C):
    """Sinusoidal function: y = A * sin(2*pi*f*t + phi) + C."""
    return A * np.sin(2 * np.pi * f * t + phi) + C


def fit_sin_fft(C):
    """Fit sinusoidal functions on temporal SVD coefficients using FFT initialization.

    Args:
        C: array [T, rank] — temporal coefficients

    Returns:
        params_list: list of (A, f, phi, C_offset) tuples, one per component
    """
    T, rank = C.shape
    t = np.arange(T, dtype=np.float64)
    params_list = []

    for i in range(rank):
        y = C[:, i].astype(np.float64)
        guess_C = np.mean(y)
        guess_A = 3 * np.std(y) / (2 ** 0.5)
        y_centered = y - guess_C
        fft_vals = np.fft.rfft(y_centered)
        fft_freqs = np.fft.rfftfreq(len(t), d=(t[1] - t[0]))
        peak_idx = np.argmax(np.abs(fft_vals[1:])) + 1
        guess_f = fft_freqs[peak_idx]

        try:
            popt, _ = curve_fit(
                _sin_func, t, y,
                p0=[guess_A, guess_f, 0.0, guess_C],
                maxfev=10000,
            )
            params_list.append(tuple(popt))
        except RuntimeError:
            # Fallback: use initial guesses directly
            params_list.append((guess_A, guess_f, 0.0, guess_C))

    return params_list


def predict_sin_fft(params_list, target_t):
    """Predict SVD coefficients at a given time index using fitted sin parameters.

    Args:
        params_list: list of (A, f, phi, C) tuples from fit_sin_fft
        target_t: float — time index to predict at

    Returns:
        c_pred: array [rank] — predicted coefficients
    """
    c_pred = np.array([
        _sin_func(target_t, *params) for params in params_list
    ], dtype=np.float32)
    return c_pred


def fit_polynomial(C, degree):
    """Fit a polynomial of the given degree on each temporal SVD coefficient.

    Args:
        C: array [T, rank] — temporal coefficients
        degree: int — polynomial degree (degree=1 ≡ fit_linear)

    Returns:
        coeffs_list: list of np.poly1d objects, one per rank component.
                     poly1d(target_t) yields the predicted coefficient.
    """
    T, rank = C.shape
    t = np.arange(T, dtype=np.float64)
    coeffs_list = []
    for i in range(rank):
        y = C[:, i].astype(np.float64)
        c = np.polyfit(t, y, degree)
        coeffs_list.append(np.poly1d(c))
    return coeffs_list


def predict_polynomial(coeffs_list, target_t):
    """Predict SVD coefficients at a given time index using fitted polynomials.

    Args:
        coeffs_list: list of np.poly1d objects from fit_polynomial
        target_t: float — time index to predict at

    Returns:
        c_pred: array [rank] — predicted coefficients
    """
    c_pred = np.array(
        [float(p(target_t)) for p in coeffs_list],
        dtype=np.float32,
    )
    return c_pred


def main():
    parser = argparse.ArgumentParser(description="SVD-based low-rank extrapolation")
    parser.add_argument("--delta_dir", type=str, required=True)
    parser.add_argument("--delta_dir2", type=str, default=None,
                        help="DEPRECATED alias for --extra_delta_dirs (single dir). "
                             "Optional second delta dir whose timesteps continue after delta_dir.")
    parser.add_argument("--extra_delta_dirs", type=str, default=None,
                        help="Comma-separated list of additional delta dirs whose timesteps continue after delta_dir. "
                             "All must track the same tensor set; their `steps` lists are concatenated in order. "
                             "Each metadata.json must have a `params` dict. "
                             "Example: phase-2 + phase-3 + phase-4 deltas.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--history_steps", type=int, required=True,
                        help="Cutoff: use steps 1..history_steps to compute SVD basis")
    parser.add_argument("--rank", type=int, required=True,
                        help="SVD rank r (number of singular vectors to keep)")
    parser.add_argument("--target_steps", type=str, required=True,
                        help="Comma-separated target steps (paper-facing labels). "
                             "Internally multiplied by --target_step_scaling_factor before fitting.")
    parser.add_argument("--target_step_scaling_factor", type=float, default=None,
                        help="(Advanced) Multiplier applied to each --target_steps value before extrapolation. "
                             "If unset, inferred from --base_model (Qwen2.5-Math-1.5B → 2.0; others → 1.0). "
                             "Output directory names always use the user-passed target_steps so they line up with the paper.")
    parser.add_argument("--mode", type=str, required=True, choices=["reconstruct", "fit_reconstruct", "extrapolate"],
                        help="reconstruct: exact low-rank projection at known steps (no fitting); "
                             "fit_reconstruct: fit temporal function on all history coefficients, predict at known steps (oracle subspace sanity check); "
                             "extrapolate: fit temporal function on history coefficients, predict at future steps")
    parser.add_argument("--fit_method", type=str, default="linear",
                        choices=["linear", "sin_fft", "polynomial"],
                        help="Fitting method for temporal coefficients: "
                             "linear = y=a*t+b; sin_fft = A*sin(2*pi*f*t+phi)+C with FFT init; "
                             "polynomial = degree-`--degree` polynomial fit (default: linear)")
    parser.add_argument("--degree", type=int, default=2,
                        help="Polynomial degree when --fit_method=polynomial (default: 2)")
    parser.add_argument("--window", type=int, default=-1,
                        help="Sliding window size for fit (-1 = use all history, default)")
    parser.add_argument("--future_extrapolation_window", type=int, default=-1,
                        help="Autoregressive re-fit interval: predict F steps, slide window, re-fit. "
                             "-1 = fit once and predict all targets directly (default). "
                             "1 = re-fit before every predicted step (fully autoregressive).")
    parser.add_argument("--svd_cache_dir", type=str, default=None,
                        help="Directory to cache/load SVD results (V_r, S, C, explained_var) per tensor. "
                             "If set and cache exists, SVD computation is skipped. "
                             "Cache is keyed by delta_dir, history_steps, and rank.")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    delta_dir = Path(args.delta_dir)
    # Build extra_dirs list: --extra_delta_dirs (comma-list) takes precedence; --delta_dir2 is back-compat alias
    if args.extra_delta_dirs:
        extra_dirs = [Path(d.strip()) for d in args.extra_delta_dirs.split(",") if d.strip()]
    elif args.delta_dir2:
        extra_dirs = [Path(args.delta_dir2)]
    else:
        extra_dirs = []
    output_dir = Path(args.output_dir)
    target_steps = [int(s) for s in args.target_steps.split(",")]  # paper-facing labels; used for output dir naming
    # Infer scaling factor from --base_model when not explicitly set.
    # Qwen2.5-Math-1.5B was trained with a checkpoint cadence that produces one saved ckpt per 2 optimizer steps,
    # so paper-facing step labels are half the fit-axis index.
    def _infer_scaling(base_model: str) -> float:
        return 2.0 if "Qwen2.5-Math-1.5B" in base_model else 1.0
    _scaling = float(args.target_step_scaling_factor) if args.target_step_scaling_factor is not None else _infer_scaling(args.base_model)
    def _eff(label: int) -> int:
        """Map a paper-facing target step to the effective time-axis index used during fitting."""
        return int(round(label * _scaling))
    if _scaling != 1.0:
        if args.mode != "extrapolate":
            raise SystemExit(f"--target_step_scaling_factor != 1.0 is only supported with --mode extrapolate (got {args.mode!r}).")
        if args.future_extrapolation_window != -1:
            raise SystemExit("--target_step_scaling_factor != 1.0 is only supported with --future_extrapolation_window=-1 (non-autoregressive).")
        print(f"[scaling] target_step_scaling_factor={_scaling}: paper-step labels {target_steps} "
              f"→ effective fit-axis steps {[_eff(s) for s in target_steps]}")

    # Load metadata
    with open(delta_dir / "metadata.json") as f:
        metadata = json.load(f)
    steps = list(metadata["steps"])
    params = metadata["params"]
    n_timesteps = metadata["n_timesteps"]

    # Deltas are stored on disk in float16 by default (release convention).
    # `metadata.json` may include a `"dtype"` field for forward compatibility.
    _dtype_str = metadata.get("dtype", "float16")
    delta_dtype = np.dtype(_dtype_str).type
    print(f"Loading deltas with dtype={_dtype_str}")

    # phase_meta: ordered list of (Path, params_dict, n_timesteps) per dir, including delta_dir as phase 0
    phase_meta = [(delta_dir, params, n_timesteps)]

    for ed in extra_dirs:
        with open(ed / "metadata.json") as f:
            md = json.load(f)
        if "params" not in md:
            raise ValueError(
                f"{ed}/metadata.json has no `params` dict. "
                "Run a retrofit script to add it (mirroring delta_dir's schema)."
            )
        # Validate compatible tensor sets and shapes
        if set(params.keys()) != set(md["params"].keys()):
            raise ValueError(f"{ed} tracks different tensor names than {delta_dir}")
        for k, info in params.items():
            if list(info["shape"]) != list(md["params"][k]["shape"]):
                raise ValueError(f"shape mismatch on {k} in {ed}: {info['shape']} vs {md['params'][k]['shape']}")
        # Disallow step overlap so step_to_idx remains injective
        steps_ed = list(md["steps"])
        overlap = set(steps) & set(steps_ed)
        if overlap:
            raise ValueError(f"steps overlap between {delta_dir} and {ed}: {sorted(overlap)[:5]}...")
        steps = steps + steps_ed
        n_timesteps += md["n_timesteps"]
        phase_meta.append((ed, md["params"], md["n_timesteps"]))

    if len(phase_meta) > 1:
        print(f"Multi-dir mode: {len(phase_meta)} delta dirs, {n_timesteps} total steps")
        for p, _, n in phase_meta:
            print(f"  {p}: {n} steps")

    step_to_idx = {s: i for i, s in enumerate(steps)}

    # Build history indices
    history_indices = [i for i, s in enumerate(steps) if s <= args.history_steps]
    T = len(history_indices)
    last_history_step = steps[history_indices[-1]]
    hist_idx_set = set(history_indices)
    hist_idx_to_pos = {hi: pos for pos, hi in enumerate(history_indices)}

    print(f"Mode: {args.mode}")
    print(f"Delta dir: {delta_dir}")
    print(f"Base model: {args.base_model}")
    print(f"Output dir: {output_dir}")
    print(f"History steps: {args.history_steps} ({T} timesteps, last={last_history_step})")
    print(f"Rank: {args.rank}")
    print(f"Fit method: {args.fit_method}")
    print(f"Window: {args.window} ({'all history' if args.window == -1 else f'last {args.window} timesteps'})")
    print(f"Future extrapolation window: {args.future_extrapolation_window} ({'fit once, predict all' if args.future_extrapolation_window == -1 else f'autoregressive, re-fit every {args.future_extrapolation_window} step(s)'})")
    print(f"Target steps: {target_steps}")
    svd_cache_dir = Path(args.svd_cache_dir) if args.svd_cache_dir else None
    if svd_cache_dir:
        svd_cache_dir.mkdir(parents=True, exist_ok=True)
        # Validate or create cache metadata
        cache_meta_path = svd_cache_dir / "cache_meta.json"
        current_meta = {
            "history_steps": args.history_steps,
            "rank": args.rank,
            "delta_dir": str(args.delta_dir),
            "extra_delta_dirs": [str(p) for p, _, _ in phase_meta[1:]],
        }
        if cache_meta_path.exists():
            with open(cache_meta_path) as f:
                saved_meta = json.load(f)
            # Migration: cache built with old schema may have delta_dir2; coerce to extra_delta_dirs list
            if "delta_dir2" in saved_meta and "extra_delta_dirs" not in saved_meta:
                saved_meta["extra_delta_dirs"] = [saved_meta["delta_dir2"]] if saved_meta["delta_dir2"] else []
            for key, label in [
                ("history_steps", "history_steps"),
                ("rank", "rank"),
                ("delta_dir", "delta_dir"),
                ("extra_delta_dirs", "extra_delta_dirs"),
            ]:
                if saved_meta.get(key) != current_meta[key]:
                    raise ValueError(
                        f"SVD cache mismatch: cache was built with {label}={saved_meta.get(key)!r}, "
                        f"but current run uses {label}={current_meta[key]!r}. "
                        f"Use a different --svd_cache_dir or delete {svd_cache_dir}."
                    )
        else:
            with open(cache_meta_path, "w") as f:
                json.dump(current_meta, f, indent=2)
        print(f"SVD cache dir: {svd_cache_dir}")
    print()

    # Validate target steps
    if args.mode in ("reconstruct", "fit_reconstruct"):
        for t in target_steps:
            if t not in step_to_idx:
                raise ValueError(f"Step {t} not in metadata ({args.mode} requires known steps)")
    elif args.mode == "extrapolate":
        for t in target_steps:
            if t <= last_history_step:
                print(f"Warning: target step {t} <= last history step {last_history_step}")

    # Load base model
    print(f"Loading base model {args.base_model}...")
    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    theta_0 = {k: v.float() for k, v in model.state_dict().items()}
    print(f"Base model loaded ({len(theta_0)} tensors)\n")

    # Skip lm_head ONLY if the model ties it to embed_tokens. For untied models
    # (e.g., GLM-4), lm_head must be treated as an independent tracked tensor.
    _is_tied_svd = getattr(config, "tie_word_embeddings", True)
    skip_params = set()
    if _is_tied_svd and "lm_head.weight" in params and "model.embed_tokens.weight" in params:
        skip_params.add("lm_head.weight")
    # Loud assertion for untied models: lm_head delta must be present.
    if (not _is_tied_svd) and ("lm_head.weight" not in params):
        raise AssertionError(
            "Untied model (tie_word_embeddings=False) but no lm_head.weight delta found. "
            "Re-run precompute_deltas.py on the updated code to include lm_head."
        )

    # Initialize predictions storage
    predictions = {step: {} for step in target_steps}

    # Process each tensor
    tensor_names = [n for n in params if n not in skip_params]
    variance_report = []

    for tensor_name in tqdm(tensor_names, desc="Processing tensors"):
        info = params[tensor_name]
        shape = tuple(info["shape"])
        numel = info["numel"]
        if len(phase_meta) > 1:
            filepath = [(p / pp[tensor_name]["filename"], n) for (p, pp, n) in phase_meta]
        else:
            filepath = delta_dir / info["filename"]
        dtype_np = delta_dtype  # deltas stored on disk in float16 (override via metadata.json `dtype` field)

        # Check SVD cache
        safe_name = tensor_name.replace(".", "_")
        cache_file = svd_cache_dir / f"{safe_name}.npz" if svd_cache_dir else None

        MAX_MATRIX_BYTES = 200 * 1024**3

        if cache_file and cache_file.exists():
            cached = np.load(cache_file)
            S = cached["S"]
            explained_var = float(cached["explained_var"])
            if "V_r" in cached:
                # Normal cache: V_r stored
                V_r = cached["V_r"]
                U_r = None
                vr_too_large = False
                needs_C = args.mode in ("extrapolate", "fit_reconstruct")
                if needs_C and "C" in cached:
                    C_cached = cached["C"]
                elif needs_C:
                    # Cache built in reconstruct mode — compute C and update cache
                    tqdm.write(f"  {tensor_name}: loaded V_r from cache, computing C...")
                    C_cached = np.zeros((T, args.rank), dtype=np.float32)
                    for i, hist_idx in enumerate(history_indices):
                        row_i = load_delta_row(filepath, shape, hist_idx, dtype_np)
                        C_cached[i] = row_i @ V_r
                    np.savez(cache_file, V_r=V_r, S=S, explained_var=np.array([explained_var]), C=C_cached)
                    tqdm.write(f"  {tensor_name}: updated cache with C")
                else:
                    C_cached = None
            else:
                # V_r-free cache: U_r stored (V_r was too large)
                U_r = cached["U_r"]
                V_r = None
                vr_too_large = True
                C_cached = (U_r * S[np.newaxis, :]).astype(np.float32) if args.mode in ("extrapolate", "fit_reconstruct") else None
            tqdm.write(f"  {tensor_name}: loaded SVD from cache")
        else:
            matrix_bytes = T * numel * 4  # float32
            vr_bytes = numel * min(args.rank, T) * 4

            if matrix_bytes <= MAX_MATRIX_BYTES:
                # Full matrix fits — standard in-memory SVD
                M = np.zeros((T, numel), dtype=np.float32)
                for i, hist_idx in enumerate(history_indices):
                    M[i] = load_delta_row(filepath, shape, hist_idx, dtype_np)
                V_r, S, U_r, explained_var = compute_svd(M, args.rank)
                vr_too_large = False
                del M
            elif vr_bytes <= MAX_MATRIX_BYTES:
                # Matrix too large but V_r fits — streaming SVD
                tqdm.write(f"  {tensor_name}: streaming SVD ({matrix_bytes / 1e9:.1f} GB matrix)")
                V_r, S, U_r, explained_var = compute_svd_streaming(
                    filepath, numel, history_indices, args.rank, dtype_np
                )
                vr_too_large = False
            else:
                # Both matrix and V_r too large — Gram-only, no V_r stored
                tqdm.write(f"  {tensor_name}: gram-only streaming (V_r would be {vr_bytes / 1e9:.1f} GB)")
                U_r, S, explained_var = compute_gram_streaming(
                    filepath, numel, history_indices, args.rank, dtype_np
                )
                V_r = None
                vr_too_large = True

            # Compute C analytically (C = U_r * S) or via projection
            if args.mode in ("extrapolate", "fit_reconstruct"):
                if vr_too_large:
                    # C = U_r * S is exact: derived from C = M @ V_r = M @ M^T @ U_r / S = G @ U_r / S = U_r * S
                    C_cached = (U_r * S[np.newaxis, :]).astype(np.float32)
                else:
                    C_cached = np.zeros((T, args.rank), dtype=np.float32)
                    for i, hist_idx in enumerate(history_indices):
                        row_i = load_delta_row(filepath, shape, hist_idx, dtype_np)
                        C_cached[i] = row_i @ V_r
            else:
                C_cached = None

            if cache_file:
                if vr_too_large:
                    save_dict = dict(U_r=U_r, S=S, explained_var=np.array([explained_var]))
                else:
                    save_dict = dict(V_r=V_r, S=S, explained_var=np.array([explained_var]))
                    if C_cached is not None:
                        save_dict["C"] = C_cached
                np.savez(cache_file, **save_dict)
                tqdm.write(f"  {tensor_name}: saved SVD to cache")

        variance_report.append((tensor_name, explained_var, S.tolist()))
        tqdm.write(f"  {tensor_name}: explained variance = {explained_var:.4f} (rank {len(S)})")

        if args.mode == "reconstruct":
            if vr_too_large:
                target_indices_list = [step_to_idx[step] for step in target_steps]
                all_in_history = all(ti in hist_idx_set for ti in target_indices_list)
                if all_in_history:
                    # Analytic c_targets from eigendecomposition: C = U_r * S
                    # Skips one full streaming pass over the data
                    c_targets = np.array([
                        U_r[hist_idx_to_pos[ti]] * S for ti in target_indices_list
                    ], dtype=np.float32)
                    delta_lrs = predict_streaming(
                        filepath, numel, history_indices, c_targets, U_r, S, dtype_np
                    )
                else:
                    _, delta_lrs = reconstruct_streaming(
                        filepath, numel, history_indices, target_indices_list, U_r, S, args.rank, dtype_np
                    )
                for j, step in enumerate(target_steps):
                    predictions[step][tensor_name] = delta_lrs[j].reshape(shape)
            else:
                for step in target_steps:
                    idx = step_to_idx[step]
                    if idx in hist_idx_set and U_r is not None:
                        # Target in history — use analytic coefficients (avoids loading delta row)
                        c_t = (U_r[hist_idx_to_pos[idx]] * S).astype(np.float32)
                    else:
                        delta_t = load_delta_row(filepath, shape, idx, dtype_np)  # [d]
                        c_t = delta_t @ V_r  # [rank]
                    delta_lr = (c_t @ V_r.T).astype(np.float32)  # [d]
                    predictions[step][tensor_name] = delta_lr.reshape(shape)

        elif args.mode == "fit_reconstruct":
            # Fit temporal function on all history coefficients, predict at known target steps.
            # Oracle sanity check: tests whether temporal fitting error matters in weight/eval space.
            # --window is intentionally ignored; always fits on all history (oracle setup).
            C = C_cached
            # Map each history step to its 0-based position (matches np.arange(T) in fit_linear/fit_sin_fft)
            hist_step_to_pos = {steps[history_indices[pos]]: pos for pos in range(T)}
            c_preds_list = []  # collect [rank] per target step, in target_steps order

            if args.fit_method == "linear":
                slopes, intercepts = fit_linear(C)
                for step in target_steps:
                    t_query = float(hist_step_to_pos[step])
                    c_preds_list.append((slopes * t_query + intercepts).astype(np.float32))

            elif args.fit_method == "sin_fft":
                sin_params = fit_sin_fft(C)
                for step in target_steps:
                    t_query = float(hist_step_to_pos[step])
                    c_preds_list.append(predict_sin_fft(sin_params, t_query))

            elif args.fit_method == "polynomial":
                poly_params = fit_polynomial(C, args.degree)
                for step in target_steps:
                    t_query = float(hist_step_to_pos[step])
                    c_preds_list.append(predict_polynomial(poly_params, t_query))

            c_preds_array = np.array(c_preds_list, dtype=np.float32)  # [n_targets, rank]
            if vr_too_large:
                delta_preds = predict_streaming(filepath, numel, history_indices, c_preds_array, U_r, S, dtype_np)
                for j, step in enumerate(target_steps):
                    predictions[step][tensor_name] = delta_preds[j].reshape(shape)
            else:
                for j, step in enumerate(target_steps):
                    predictions[step][tensor_name] = (c_preds_array[j] @ V_r.T).reshape(shape)

        elif args.mode == "extrapolate":
            C = C_cached  # already computed (or loaded from cache)

            W = T if args.window == -1 else min(args.window, T)
            F = args.future_extrapolation_window

            # Collect c_preds per target step for deferred streaming reconstruction
            c_preds_per_step = {}  # {step: c_pred [rank]}

            if F == -1:
                # Non-autoregressive: fit once on last W history steps, predict all targets
                C_fit = C[-W:]

                if args.fit_method == "linear":
                    slopes, intercepts = fit_linear(C_fit)
                    for step in target_steps:
                        steps_ahead = _eff(step) - last_history_step
                        target_t = W - 1 + steps_ahead
                        c_preds_per_step[step] = (slopes * target_t + intercepts).astype(np.float32)

                elif args.fit_method == "sin_fft":
                    sin_params = fit_sin_fft(C_fit)
                    for step in target_steps:
                        steps_ahead = _eff(step) - last_history_step
                        target_t = W - 1 + steps_ahead
                        c_preds_per_step[step] = predict_sin_fft(sin_params, target_t)

                elif args.fit_method == "polynomial":
                    poly_params = fit_polynomial(C_fit, args.degree)
                    for step in target_steps:
                        steps_ahead = _eff(step) - last_history_step
                        target_t = W - 1 + steps_ahead
                        c_preds_per_step[step] = predict_polynomial(poly_params, target_t)

            else:
                # Autoregressive: slide window forward, re-fitting every F steps
                from collections import deque
                window_buf = deque(C[-W:], maxlen=W)
                target_set = set(target_steps)
                steps_since_refit = F  # force fit on first step
                fit_state = None

                for next_step in range(last_history_step + 1, max(target_steps) + 1):
                    if steps_since_refit >= F:
                        C_win = np.array(window_buf, dtype=np.float32)
                        if args.fit_method == "linear":
                            fit_state = fit_linear(C_win)
                        elif args.fit_method == "sin_fft":
                            fit_state = fit_sin_fft(C_win)
                        elif args.fit_method == "polynomial":
                            fit_state = fit_polynomial(C_win, args.degree)
                        steps_since_refit = 0

                    target_t = W - 1 + 1
                    if args.fit_method == "linear":
                        slopes, intercepts = fit_state
                        c_pred = (slopes * target_t + intercepts).astype(np.float32)
                    elif args.fit_method == "sin_fft":
                        c_pred = predict_sin_fft(fit_state, target_t)
                    elif args.fit_method == "polynomial":
                        c_pred = predict_polynomial(fit_state, target_t)

                    if next_step in target_set:
                        c_preds_per_step[next_step] = c_pred

                    window_buf.append(c_pred)
                    steps_since_refit += 1

            # Reconstruct weight deltas from collected c_preds
            if vr_too_large:
                c_preds_array = np.array([c_preds_per_step[s] for s in target_steps], dtype=np.float32)
                delta_preds = predict_streaming(filepath, numel, history_indices, c_preds_array, U_r, S, dtype_np)
                for j, step in enumerate(target_steps):
                    predictions[step][tensor_name] = delta_preds[j].reshape(shape)
            else:
                for step in target_steps:
                    predictions[step][tensor_name] = (c_preds_per_step[step] @ V_r.T).reshape(shape)

        if not vr_too_large:
            del V_r

    # Print variance summary
    print(f"\n{'='*60}")
    print(f"Explained variance summary (rank={args.rank}):")
    print(f"{'='*60}")
    variances = [v for _, v, _ in variance_report]
    print(f"  Mean: {np.mean(variances):.4f}")
    print(f"  Min:  {np.min(variances):.4f} ({variance_report[np.argmin(variances)][0]})")
    print(f"  Max:  {np.max(variances):.4f} ({variance_report[np.argmax(variances)][0]})")
    print(f"  Median: {np.median(variances):.4f}")
    n_above_90 = sum(1 for v in variances if v >= 0.90)
    n_above_95 = sum(1 for v in variances if v >= 0.95)
    n_above_99 = sum(1 for v in variances if v >= 0.99)
    print(f"  >=90%: {n_above_90}/{len(variances)}")
    print(f"  >=95%: {n_above_95}/{len(variances)}")
    print(f"  >=99%: {n_above_99}/{len(variances)}")

    # Save variance report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"variance_report_h{args.history_steps}_r{args.rank}.json"
    with open(report_path, "w") as f:
        json.dump({
            "history_steps": args.history_steps,
            "rank": args.rank,
            "fit_method": args.fit_method,
            "degree": args.degree if args.fit_method == "polynomial" else None,
            "window": args.window,
            "mode": args.mode,
            "summary": {
                "mean": float(np.mean(variances)),
                "min": float(np.min(variances)),
                "max": float(np.max(variances)),
                "median": float(np.median(variances)),
                "above_90pct": n_above_90,
                "above_95pct": n_above_95,
                "above_99pct": n_above_99,
                "total_tensors": len(variances),
            },
            "per_tensor": [
                {"name": name, "explained_variance": float(ev), "singular_values": sv}
                for name, ev, sv in variance_report
            ],
        }, f, indent=2)
    print(f"\nVariance report saved to {report_path}")

    # Save checkpoints
    print(f"\nSaving {len(target_steps)} checkpoints...")
    for step in target_steps:
        print(f"\n--- Step {step} ---")

        # Build predicted state dict
        predicted_state_dict = {}
        for name in theta_0:
            if name in predictions[step]:
                delta = torch.from_numpy(predictions[step][name]).float()
                result = theta_0[name] + delta
                predicted_state_dict[name] = result.to(torch.bfloat16)
            elif _is_tied_svd and name == "lm_head.weight" and "model.embed_tokens.weight" in predictions[step]:
                # Tied model: reuse embed_tokens delta for lm_head.
                delta = torch.from_numpy(predictions[step]["model.embed_tokens.weight"]).float()
                result = theta_0["model.embed_tokens.weight"] + delta
                predicted_state_dict[name] = result.to(torch.bfloat16)
            else:
                predicted_state_dict[name] = theta_0[name].to(torch.bfloat16)

        # Handle tied weights: for TIED models only, force lm_head == embed_tokens after reconstruction.
        if _is_tied_svd and "model.embed_tokens.weight" in predicted_state_dict and "lm_head.weight" in predicted_state_dict:
            predicted_state_dict["lm_head.weight"] = predicted_state_dict["model.embed_tokens.weight"].clone()

        step_dir = output_dir / f"step{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        model.load_state_dict(predicted_state_dict)
        config.save_pretrained(step_dir)
        model.save_pretrained(step_dir, safe_serialization=True)

        del predicted_state_dict
        del predictions[step]

        size_gb = sum(f.stat().st_size for f in step_dir.iterdir()) / 1e9
        print(f"  Saved to {step_dir} ({size_gb:.1f} GB)")

    print(f"\nDone! Generated {len(target_steps)} checkpoints in {args.mode} mode.")


if __name__ == "__main__":
    main()
