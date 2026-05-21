#!/bin/bash
# End-to-end RELEX reproduction on Qwen2.5-Math-1.5B.
#
# Runs Steps 0-3 sequentially:
#   0. Download RLVR step_1..step_75 from relex-rlvr/RLVR-Qwen2.5-Math-1.5B.
#   1. Compute weight deltas Δ_t from those checkpoints.
#   2. Fit rank-1 + linear → predicted checkpoint at the target step.
#   3. Evaluate on MATH + 5 OOD benchmarks.

set -e

export HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-Math-1.5B}
HUB_REPO=${HUB_REPO:-relex-rlvr/RLVR-Qwen2.5-Math-1.5B}
CKPT_DIR=${CKPT_DIR:-$PWD/rlvr_traj/Qwen2.5-Math-1.5B}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DELTA_DIR=$ROOT/outputs/deltas/Qwen2.5-Math-1.5B
RELEX_DIR=$ROOT/outputs/relex/Qwen2.5-Math-1.5B/cutoff75
SVD_CACHE=$ROOT/outputs/svd_cache/Qwen2.5-Math-1.5B/cutoff75
EVAL_DIR=$ROOT/eval_results/Qwen2.5-Math-1.5B-RELEX-c75-step500

T_CUT=75
TARGET_STEP=500

# ---------------------------------------------------------------------------
# Step 0 — Download RLVR step_1..step_T_CUT branches from the Hub.
# ---------------------------------------------------------------------------
echo "=== Step 0/3: download RLVR steps 1..$T_CUT from $HUB_REPO ==="
python $ROOT/scripts/download_rlvr_trajectory.py \
    --hub_repo $HUB_REPO \
    --num_steps $T_CUT \
    --output_dir $CKPT_DIR

# ---------------------------------------------------------------------------
# Step 1 — Precompute per-step deltas.
# ---------------------------------------------------------------------------
echo "=== Step 1/3: precompute deltas (history_steps window) ==="
python $ROOT/scripts/precompute_deltas.py \
    --base_model $BASE_MODEL \
    --checkpoint_dir $CKPT_DIR \
    --output_dir $DELTA_DIR \
    --max_checkpoints $T_CUT \
    --num_workers 8

# ---------------------------------------------------------------------------
# Step 2 — RELEX: rank-1 SVD + linear extrapolation → predicted checkpoint.
# ---------------------------------------------------------------------------
echo "=== Step 2/3: build RELEX predicted checkpoint (T_cut=$T_CUT, target=$TARGET_STEP) ==="
python $ROOT/scripts/svd_extrapolation.py \
    --base_model $BASE_MODEL \
    --delta_dir $DELTA_DIR \
    --history_steps $T_CUT \
    --rank 1 \
    --fit_method linear \
    --mode extrapolate \
    --target_steps $TARGET_STEP \
    --output_dir $RELEX_DIR \
    --svd_cache_dir $SVD_CACHE

PRED_CKPT=$RELEX_DIR/step$TARGET_STEP

# ---------------------------------------------------------------------------
# Step 3 — Evaluate.
# ---------------------------------------------------------------------------
echo "=== Step 3/3: evaluate ==="

# 3a — MATH
python $ROOT/scripts/eval.py \
    --model_name $PRED_CKPT \
    --datasets TianHongZXY/MATH \
    --output_dir $EVAL_DIR \
    --temperature 0.0 --top_p 1.0 --top_k -1 \
    --max_tokens 4096 --num_generation 1 \
    --batch_size 1000

# 3b — OOD benchmarks
for DS in TianHongZXY/AIME2025 MathArena/aime_2026 \
          MathArena/hmmt_feb_2025 TianHongZXY/amc23 \
          zwhe99/simplerl-OlympiadBench; do
  python $ROOT/scripts/eval.py \
      --model_name $PRED_CKPT \
      --datasets $DS \
      --output_dir $EVAL_DIR \
      --temperature 0.7 --top_p 0.8 --top_k -1 \
      --max_tokens 4096 --num_generation 8 \
      --batch_size 1000
done

echo "=== Done. Results in $EVAL_DIR ==="
ls -la $EVAL_DIR
