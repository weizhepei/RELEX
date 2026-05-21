"""Download an RLVR checkpoint trajectory (step_1..step_N) from a relex-rlvr Hub repo.

Each step is a separate branch on the Hub; this script materializes them as local
per-step directories named `global_step_<N>/`, suitable for input to
`scripts/precompute_deltas.py --checkpoint_dir`.

Idempotent — skips steps already present.
"""

import argparse
import os
from huggingface_hub import snapshot_download

ALLOW = ["*.safetensors", "*.json", "*.txt", "*.jinja"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub_repo", required=True,
                        help="HF Hub repo id, e.g. relex-rlvr/RLVR-Qwen2.5-Math-1.5B")
    parser.add_argument("--num_steps", type=int, required=True,
                        help="Download branches step_1..step_<num_steps>")
    parser.add_argument("--output_dir", required=True,
                        help="Local root; each step goes to <output_dir>/global_step_<N>/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for n in range(1, args.num_steps + 1):
        dst = os.path.join(args.output_dir, f"global_step_{n}")
        if os.path.exists(os.path.join(dst, "config.json")):
            continue
        print(f"[{n}/{args.num_steps}] {args.hub_repo}@step_{n} -> {dst}", flush=True)
        snapshot_download(
            args.hub_repo,
            revision=f"step_{n}",
            local_dir=dst,
            allow_patterns=ALLOW,
        )
    print(f"Done. {args.num_steps} step directories at {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
