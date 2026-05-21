# RELEX (REinforcement Learning EXtrapolation)

[![arXiv](https://img.shields.io/badge/arXiv-2605.21468-b31b1b.svg)](https://arxiv.org/abs/2605.21468)
[![Hugging Face](https://img.shields.io/badge/🤗-Checkpoints-yellow)](https://huggingface.co/relex-rlvr)
[![Notion Blog](https://img.shields.io/static/v1?label=&message=Notion%20Blog&color=000000&logo=notion&logoColor=white)](https://weizhepei.notion.site/you-only-need-minimal-rlvr-training)
[![X Post](https://img.shields.io/static/v1?label=&message=X%20Post&color=000000&logo=x&logoColor=white)](https://x.com/weizhepei/status/2055034616180867536?s=20)

This repository contains the scripts needed to reproduce the paper "You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories":

1. Verify **Finding 1**: RLVR updates are extremely low-rank (rank-1) across the training trajectory.
2. Verify **Finding 2**: the rank-1 coefficient evolves linearly with training step.
3. Run the **RELEX** method end-to-end: fit a rank-1 + linear model on a partial-training prefix, extrapolate to a target step, build a predicted checkpoint, evaluate on both in-domain and out-of-domain benchmarks.

---

## Requirements

```bash
conda create -y -n relex python=3.11
conda activate relex
pip install -r requirements.txt
```

---

## Quick start

End-to-end for any backbone:

```bash
bash run_qwen2.5-math-1.5b.sh   # Qwen2.5-Math-1.5B
bash run_qwen3-4b-base.sh       # Qwen3-4B-Base
bash run_qwen3-8b-base.sh       # Qwen3-8B-Base
```

Or run the steps individually (Qwen2.5-Math-1.5B example below):

### Step 1 — Compute per-step weight deltas

First materialize the first 75 step branches of the released RLVR trajectory (enough for the RELEX pipeline; raise `--num_steps` if you want more):

```bash
python scripts/download_rlvr_trajectory.py \
    --hub_repo relex-rlvr/RLVR-Qwen2.5-Math-1.5B \
    --num_steps 75 \
    --output_dir ./rlvr_traj/Qwen2.5-Math-1.5B
```

Then compute the per-step deltas:

```bash
python scripts/precompute_deltas.py \
    --base_model Qwen/Qwen2.5-Math-1.5B \
    --checkpoint_dir ./rlvr_traj/Qwen2.5-Math-1.5B \
    --output_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --max_checkpoints 75 \
    --num_workers 8
```


### Step 2 — RELEX: rank-1 SVD + linear extrapolation + reconstruct target step

```bash
python scripts/svd_extrapolation.py \
    --base_model Qwen/Qwen2.5-Math-1.5B \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --history_steps 75 \
    --rank 1 \
    --fit_method linear \
    --mode extrapolate \
    --target_steps 500 \
    --output_dir outputs/relex/Qwen2.5-Math-1.5B/cutoff75 \
    --svd_cache_dir outputs/svd_cache/Qwen2.5-Math-1.5B/cutoff75
```

Note `--svd_cache_dir` is **strongly recommended** — it lets later calls with different `--target_steps` reuse the SVD cache without recomputation.

---

## Analysis and Evaluation

### Verify Finding 1: reconstruction with rank-1 subspace

Tests whether the weight dynamics are truly low-rank — projects each observed delta into the rank-r subspace and back, with no temporal fitting. If a reconstructed checkpoint's eval accuracy matches the original, the rank-r subspace captures the essential training signal.

> **Note.** Reconstruction over the full 500-step trajectory needs all `step_1`..`step_500` branches of [`relex-rlvr/RLVR-Qwen2.5-Math-1.5B`](https://huggingface.co/relex-rlvr/RLVR-Qwen2.5-Math-1.5B). Re-run Step 1 with both `--num_steps 500` (download script) and `--max_checkpoints 500` (precompute_deltas) to materialize the full trajectory and its deltas locally.

```bash
python scripts/svd_extrapolation.py \
    --base_model Qwen/Qwen2.5-Math-1.5B \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --history_steps 500 --rank 1 \
    --target_steps 100,200,300,400,500 \
    --mode reconstruct \
    --output_dir outputs/reconstruct/Qwen2.5-Math-1.5B/rank1
```

### Verify Finding 2: rank-1 coefficient linearity

```bash
python scripts/plot_svd_coefficients.py \
    --delta_dir outputs/deltas/Qwen2.5-Math-1.5B \
    --history_steps 75 --rank 1 \
    --output_dir plots/svd_coefficients/Qwen2.5-Math-1.5B
```

### Evaluation

In-domain (MATH):

```bash
python scripts/eval.py \
    --model_name relex-rlvr/RELEX-Qwen2.5-Math-1.5B \
    --datasets TianHongZXY/MATH \
    --output_dir eval_results/RELEX-Qwen2.5-Math-1.5B \
    --temperature 0.0 --top_p 1.0 --top_k -1 \
    --max_tokens 4096 --num_generation 1 \
    --batch_size 1000
```

Out-of-domain (AIME25, AIME26, HMMT25, AMC23, OlympiadBench):

```bash
for DS in TianHongZXY/AIME2025 MathArena/aime_2026 \
          MathArena/hmmt_feb_2025 TianHongZXY/amc23 \
          zwhe99/simplerl-OlympiadBench; do
  python scripts/eval.py \
      --model_name relex-rlvr/RELEX-Qwen2.5-Math-1.5B \
      --datasets $DS \
      --output_dir eval_results/RELEX-Qwen2.5-Math-1.5B \
      --temperature 0.7 --top_p 0.8 --top_k -1 \
      --max_tokens 4096 --num_generation 8 \
      --batch_size 1000
done
```

---

## Released checkpoints

All paper artifacts are released on the [🤗 relex-rlvr](https://huggingface.co/relex-rlvr) Hub organization:

| Repo | Contents |
|---|---|
| [`RLVR-Qwen2.5-Math-1.5B`](https://huggingface.co/relex-rlvr/RLVR-Qwen2.5-Math-1.5B) | RLVR trajectory: `main` = step-500, branches `step_1`..`step_500` |
| [`RLVR-Qwen3-4B-Base`](https://huggingface.co/relex-rlvr/RLVR-Qwen3-4B-Base) | RLVR trajectory: `main` = step-500, branches `step_1`..`step_75` |
| [`RLVR-Qwen3-8B-Base`](https://huggingface.co/relex-rlvr/RLVR-Qwen3-8B-Base) | RLVR trajectory: `main` = step-500, branches `step_1`..`step_100` |
| [`RELEX-Qwen2.5-Math-1.5B`](https://huggingface.co/relex-rlvr/RELEX-Qwen2.5-Math-1.5B) | RELEX prediction, T_cut=75, target step=500 |
| [`RELEX-Qwen3-4B-Base`](https://huggingface.co/relex-rlvr/RELEX-Qwen3-4B-Base) | RELEX prediction, T_cut=75, target step=500 |
| [`RELEX-Qwen3-8B-Base`](https://huggingface.co/relex-rlvr/RELEX-Qwen3-8B-Base) | RELEX prediction, T_cut=100, target step=750 |

Trajectory branches use the `revision="step_N"` convention, so you can load any intermediate RLVR checkpoint without downloading the entire trajectory:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "relex-rlvr/RLVR-Qwen2.5-Math-1.5B",
    revision="step_50",   # any step in [1, 500]; omit revision to get final step-500
)
```

---

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zhepei (zhepei.wei@virginia.edu). If you encounter any problems when using the code, or want to report a bug, feel free to open an issue! Please try to specify the problem with details so we can help you better and quicker!

## Citation

If you find this codebase useful, please consider citing:

```bibtex
@article{wei2026relex,
  title   = {You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories},
  author  = {Wei, Zhepei and Zhu, Xinyu and Chen, Wei-Lin and Huang, Chengsong and Huang, Jiaxin and Meng, Yu},
  journal = {arXiv preprint arXiv:2605.21468},
  year    = {2026}
}
```

