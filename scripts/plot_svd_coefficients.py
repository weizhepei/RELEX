"""
Visualize SVD temporal coefficients (C = U_r S_r) to check if they are linear.

For each selected tensor, loads all available delta steps, computes SVD on the
history window, and plots the temporal coefficients with linear fit overlaid.
Crucially, all 500 oracle steps are shown as scatter points — the first
history_steps are used to fit the linear function, the remaining steps are
out-of-sample and reveal how well the fit extrapolates.

Two modes (--oracle_subspace):
  Option 1 (default): V_r computed from history steps only [T_hist, d].
    - Realistic: matches what the actual extrapolation algorithm uses.
    - Question: does the subspace we use for extrapolation have linear dynamics?

  Option 2 (--oracle_subspace): V_r computed from ALL 500 steps [T_full, d].
    - Cheating: uses future information unavailable at prediction time.
    - Question: is the true low-rank structure of the full trajectory linear?
    - If Option 2 looks linear but Option 1 doesn't, the subspace itself drifts.

Note on "oracle" out-of-sample points:
  The orange scatter points (steps 101-500) are computed as:
    C_full[t] = M_full[t] @ V_r   (true delta projected onto history subspace)
  M_full[t] contains the ground-truth delta from real training, so the input is
  oracle. However, the projection onto V_r (computed from steps 1-100) may lose
  information if the subspace drifts over time — i.e. if new principal directions
  emerge at later steps that weren't present in the first 100.
  Therefore what the orange points show is: the oracle delta as seen through the
  history subspace lens. This is exactly what matters for evaluating our algorithm,
  since our extrapolation also operates in this same V_r space.
  Interpreting the plot:
    - Orange follows the linear fit  →  V_r is stable AND dynamics are linear
    - Orange deviates from the fit   →  subspace drifted, or dynamics are non-linear
    - Option 2 linear but Option 1 not  →  problem is subspace drift, not non-linearity

No base model needed — only reads delta files.

Usage:
# Option 1 (realistic subspace, default):
python plot_svd_coefficients.py \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --history_steps 100 --rank 5 \
    --output_dir plots/svd_coefficients/Qwen2.5-Math-1.5B

# Option 2 (oracle subspace, uses ALL steps):
python plot_svd_coefficients.py \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --history_steps 100 --rank 5 --oracle_subspace \
    --output_dir plots/svd_coefficients/Qwen2.5-Math-1.5B-oracle
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_delta_row(filepath, shape, idx, dtype=np.float16):
    numel = int(np.prod(shape))
    offset = idx * numel * np.dtype(dtype).itemsize
    return np.fromfile(filepath, dtype=dtype, count=numel, offset=offset)


def load_delta_row_slice(filepath, numel, idx, start, end, dtype=np.float16):
    itemsize = np.dtype(dtype).itemsize
    offset = (idx * numel + start) * itemsize
    return np.fromfile(filepath, dtype=dtype, count=end - start, offset=offset)


def compute_svd_streaming(filepath, shape, all_indices, svd_indices, rank, dtype=np.float16, chunk_d=100_000_000):
    """Streaming SVD + C_full computation for tensors too large to fit in memory.

    Args:
        all_indices: row indices for all steps (to compute C_full)
        svd_indices: row indices used to compute the SVD subspace (history or all)
    Returns:
        V_r [d, rank], S [rank], explained_var, C_full [len(all_indices), rank]
    """
    numel = int(np.prod(shape))
    T_svd = len(svd_indices)
    T_all = len(all_indices)
    rank = min(rank, T_svd)

    # Pass 1: Gram matrix via d-chunks (compute in `dtype` (default fp16, read from metadata)
    G = np.zeros((T_svd, T_svd), dtype=np.float32)
    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        M_chunk = np.zeros((T_svd, d_end - d_start), dtype=np.float32)
        for i, hi in enumerate(svd_indices):
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
    U_r = eigenvectors[:, :rank]
    S = np.sqrt(np.maximum(eigenvalues, 0))

    # Pass 2: V_r via d-chunks (compute in dtype)
    V_r = np.zeros((numel, rank), dtype=np.float32)
    for d_start in range(0, numel, chunk_d):
        d_end = min(d_start + chunk_d, numel)
        M_chunk = np.zeros((T_svd, d_end - d_start), dtype=np.float32)
        for i, hi in enumerate(svd_indices):
            M_chunk[i] = load_delta_row_slice(filepath, numel, hi, d_start, d_end, dtype)
        V_chunk = M_chunk.T @ U_r
        for k in range(rank):
            V_chunk[:, k] = V_chunk[:, k] / S[k] if S[k] > 1e-12 else 0.0
        V_r[d_start:d_end] = V_chunk
        del M_chunk, V_chunk

    # Pass 3: C_full = M_full @ V_r, one row at a time (compute in dtype)
    C_full = np.zeros((T_all, rank), dtype=np.float32)
    for i, hi in enumerate(all_indices):
        row = load_delta_row(filepath, shape, hi, dtype)
        C_full[i] = row @ V_r

    return V_r, S, explained_var, C_full


def compute_svd(M, rank):
    """Compute truncated SVD via Gram matrix trick. Returns V_r [d, r], S [r], explained_var."""
    T, d = M.shape
    rank = min(rank, T)
    G = M @ M.T  # [T, T]
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    total_var = np.sum(np.maximum(eigenvalues, 0))
    top_var = np.sum(np.maximum(eigenvalues[:rank], 0))
    explained_var = top_var / total_var if total_var > 0 else 0.0
    eigenvalues = eigenvalues[:rank]
    U_r = eigenvectors[:, :rank]
    S = np.sqrt(np.maximum(eigenvalues, 0))
    V_r = M.T @ U_r  # [d, rank]
    for i in range(rank):
        if S[i] > 1e-12:
            V_r[:, i] /= S[i]
        else:
            V_r[:, i] = 0.0
    return V_r, S, explained_var


def fit_linear(t, y):
    """OLS linear fit. Returns slope, intercept, R²."""
    t_mean = t.mean()
    y_mean = y.mean()
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0, y_mean, 1.0
    slope = np.sum((t - t_mean) * (y - y_mean)) / np.sum((t - t_mean) ** 2)
    intercept = y_mean - slope * t_mean
    y_fit = slope * t + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    r2 = 1 - ss_res / ss_tot
    return slope, intercept, r2


def select_representative_tensors(params):
    """Auto-select: first/middle/last layer x q_proj/gate_proj/down_proj + embed_tokens."""
    layers = set()
    for name in params:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    layers.add(int(parts[i + 1]))
                except ValueError:
                    pass
    layers = sorted(layers)
    if not layers:
        return list(params.keys())[:6]

    selected_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    types = ["q_proj", "gate_proj", "down_proj"]
    selected = []
    for layer in selected_layers:
        for ttype in types:
            for name in params:
                if f"layers.{layer}." in name and ttype in name:
                    selected.append(name)
                    break
    for name in params:
        if "embed_tokens" in name:
            selected.append(name)
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Visualize SVD temporal coefficients")
    parser.add_argument("--delta_dir", type=str, required=True)
    parser.add_argument("--history_steps", type=int, required=True,
                        help="Steps used to fit the linear function (cutoff)")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--tensors", type=str, default=None,
                        help="Comma-separated tensor names (default: auto-select representative set)")
    parser.add_argument("--oracle_subspace", action="store_true",
                        help="Option 2: compute V_r from all steps (oracle). "
                             "Default (Option 1): V_r from history steps only.")
    parser.add_argument("--output_dir", type=str, default="plots/svd_coefficients")
    args = parser.parse_args()

    delta_dir = Path(args.delta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(delta_dir / "metadata.json") as f:
        metadata = json.load(f)
    all_steps = metadata["steps"]
    params = metadata["params"]
    dtype = np.dtype(metadata.get("dtype", "float16")).type

    # Split indices into history (fit) and future (out-of-sample)
    history_indices = [i for i, s in enumerate(all_steps) if s <= args.history_steps]
    future_indices  = [i for i, s in enumerate(all_steps) if s >  args.history_steps]
    T_hist = len(history_indices)
    T_full = len(all_steps)
    hist_steps  = [all_steps[i] for i in history_indices]
    future_steps = [all_steps[i] for i in future_indices]
    all_step_values = [all_steps[i] for i in range(T_full)]

    mode_str = "oracle_subspace" if args.oracle_subspace else "history_subspace"
    print(f"Delta dir:    {delta_dir}")
    print(f"Mode:         Option {'2 (oracle V_r from all steps)' if args.oracle_subspace else '1 (V_r from history only)'}")
    print(f"History:      {T_hist} steps ({hist_steps[0]}..{hist_steps[-1]})")
    print(f"Out-of-sample:{len(future_indices)} steps ({future_steps[0] if future_steps else 'none'}..{future_steps[-1] if future_steps else 'none'})")
    print(f"Rank:         {args.rank}")

    if args.tensors:
        selected = [t.strip() for t in args.tensors.split(",")]
    else:
        selected = select_representative_tensors(params)
    print(f"Tensors:      {len(selected)}")

    for tensor_name in selected:
        if tensor_name not in params:
            print(f"  Warning: {tensor_name} not found, skipping")
            continue

        info = params[tensor_name]
        shape = tuple(info["shape"])
        numel = info["numel"]
        filepath = delta_dir / info["filename"]

        print(f"\n  {tensor_name} ({numel:,} params)")

        MAX_MATRIX_BYTES = 200 * 1024**3
        matrix_bytes = T_full * numel * 4
        svd_indices = list(range(T_full)) if args.oracle_subspace else history_indices

        if matrix_bytes > MAX_MATRIX_BYTES:
            print(f"    Large tensor ({matrix_bytes / 1e9:.0f} GB), using streaming SVD")
            V_r, S, explained_var, C_full = compute_svd_streaming(
                filepath, shape, list(range(T_full)), svd_indices, args.rank)
        else:
            # Load full M [T_full, d] (read fp16 from disk, compute in fp32)
            M_full = np.zeros((T_full, numel), dtype=np.float32)
            for i in range(T_full):
                M_full[i] = load_delta_row(filepath, shape, i)

            svd_source = M_full if args.oracle_subspace else M_full[history_indices]
            V_r, S, explained_var = compute_svd(svd_source, args.rank)
            C_full = M_full @ V_r
            del M_full

        n_comp = C_full.shape[1]
        t_full = np.array(all_step_values, dtype=np.float64)
        t_hist = np.array(hist_steps, dtype=np.float64)

        # Fit linear on history portion only
        slopes, intercepts, r2_hist = [], [], []
        for i in range(n_comp):
            c_hist = C_full[history_indices, i].astype(np.float64)
            s, b, r2 = fit_linear(t_hist, c_hist)
            slopes.append(s)
            intercepts.append(b)
            r2_hist.append(r2)

        # R² on out-of-sample steps
        r2_oos_list = []
        for i in range(n_comp):
            if not future_indices:
                r2_oos_list.append(float('nan'))
                continue
            c_oos = C_full[future_indices, i].astype(np.float64)
            t_oos = np.array(future_steps, dtype=np.float64)
            y_pred = slopes[i] * t_oos + intercepts[i]
            ss_res = np.sum((c_oos - y_pred) ** 2)
            ss_tot = np.sum((c_oos - c_oos.mean()) ** 2)
            r2_oos_list.append(1 - ss_res / ss_tot if ss_tot > 0 else float('nan'))

        print(f"    Explained var ({mode_str}): {explained_var:.4f}")
        print(f"    R² in-sample  (top 5): {', '.join(f'{r:.3f}' for r in r2_hist[:5])}")
        print(f"    R² out-sample (top 5): {', '.join(f'{r:.3f}' for r in r2_oos_list[:5])}")

        # --- Plot ---
        n_show = min(5, n_comp)
        fig, axes = plt.subplots(n_show, 1, figsize=(14, 3.5 * n_show), sharex=True)
        if n_show == 1:
            axes = [axes]

        t_line = np.array([t_full[0], t_full[-1]])

        for i in range(n_show):
            ax = axes[i]
            var_pct = (S[i] ** 2 / np.sum(S ** 2) * 100) if np.sum(S ** 2) > 0 else 0

            # Scatter: history (blue) always; out-of-sample (orange) only in oracle mode
            ax.scatter(t_hist, C_full[history_indices, i],
                       s=18, color='steelblue', zorder=3, label='History (fit)')
            if future_indices and args.oracle_subspace:
                ax.scatter(np.array(future_steps), C_full[future_indices, i],
                           s=18, color='darkorange', zorder=3, label='Out-of-sample (oracle)')

            # Linear fit line extended over full range
            y_line = slopes[i] * t_line + intercepts[i]
            ax.plot(t_line, y_line, 'r--', linewidth=1.5,
                    label=f'Linear fit  R²_in={r2_hist[i]:.3f}  R²_oos={r2_oos_list[i]:.3f}')

            # Vertical line at history cutoff
            ax.axvline(x=args.history_steps, color='gray', linestyle=':', linewidth=1, alpha=0.7)

            ax.set_ylabel(f"Comp {i+1}\n({var_pct:.1f}%)", fontsize=10)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

        title = f"{tensor_name}  [rank={args.rank}, expl_var={explained_var:.4f}, mode={mode_str}]"
        axes[0].set_title(title, fontsize=11)
        axes[-1].set_xlabel("Training step", fontsize=11)

        plt.tight_layout()
        safe_name = tensor_name.replace(".", "_")
        path = output_dir / f"{safe_name}_{mode_str}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved {path}")

        # Save data used for plotting
        data_path = output_dir / f"{safe_name}_{mode_str}.npz"
        np.savez(data_path,
                 all_steps=np.array(all_step_values),
                 hist_steps=np.array(hist_steps),
                 future_steps=np.array(future_steps),
                 C_full=C_full,
                 history_indices=np.array(history_indices),
                 future_indices=np.array(future_indices),
                 S=S,
                 slopes=np.array(slopes),
                 intercepts=np.array(intercepts),
                 r2_hist=np.array(r2_hist),
                 r2_oos=np.array(r2_oos_list),
                 explained_var=np.array([explained_var]),
                 tensor_name=np.array([tensor_name]),
                 rank=np.array([args.rank]),
                 history_steps=np.array([args.history_steps]),
                 mode=np.array([mode_str]))
        print(f"    Saved {data_path}")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
