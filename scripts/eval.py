import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils import extract_answer_math
from grader import math_equal
os.environ["NCCL_DEBUG"] = "WARN"

def _auto_resume(output_path, num_generation, expected_total):
    """Check existing output file and determine resume point.
    
    Returns (resume_id, should_skip):
        - (0, True) if file is complete
        - (N, False) if file is partial and should resume from example N
        - (0, False) if file doesn't exist (fresh start)
    """
    if not os.path.exists(output_path):
        return 0, False

    # Count valid JSONL lines, tracking byte position of each line end
    line_positions = [0]  # position before first line
    valid_lines = 0
    with open(output_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
                valid_lines += 1
                line_positions.append(f.tell())
            except json.JSONDecodeError:
                print(f"[auto-resume] Found corrupt line {valid_lines + 1}, truncating.")
                break

    # Truncate to nearest complete example (multiple of num_generation)
    complete_lines = (valid_lines // num_generation) * num_generation
    truncate_pos = line_positions[complete_lines]
    file_size = os.path.getsize(output_path)
    if truncate_pos < file_size:
        extra = valid_lines - complete_lines
        reason = "corrupt data" if extra == 0 else f"{extra} orphan line(s) from incomplete example"
        print(f"[auto-resume] Truncating {output_path}: keeping {complete_lines}/{valid_lines} lines ({reason}).")
        with open(output_path, 'r+') as f:
            f.truncate(truncate_pos)

    expected_lines = expected_total * num_generation
    if complete_lines >= expected_lines:
        print(f"[auto-resume] {output_path} is complete ({complete_lines}/{expected_lines} lines). Skipping.")
        return 0, True

    resume_id = complete_lines // num_generation
    print(f"[auto-resume] {output_path} has {complete_lines}/{expected_lines} lines. "
          f"Resuming from example {resume_id}.")
    return resume_id, False




PROMPT_TEMPLATES = {
    "qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n",
    "glm": "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n{input}\nPlease reason step by step, and put your final answer within \\boxed{}.<|assistant|>\n",
    "llama": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{input}\nPlease reason step by step, and put your final answer within \\boxed{}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
}


def prepare_data(example, prompt_key, prompt_template):
    example['prompt'] = prompt_template.replace("{input}", example[prompt_key])
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--num_generation", type=int, default=1)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--resume_id", type=int, default=0)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--prompt_template", type=str, default="qwen", choices=list(PROMPT_TEMPLATES.keys()))
    args = parser.parse_args()
    print(args)

    # Load the model and tokenizer
    print(f"Loading model {args.model_name}")
    llm = LLM(args.model_name, tensor_parallel_size=args.num_gpus, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=args.num_generation, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    # Load the dataset
    datasets = args.datasets.split(",")
    for dataset_name in datasets:
        try:
            dataset = load_dataset(dataset_name, split=args.split)
        except ValueError as e:
            if "Unknown split" in str(e) and args.split == "test":
                print(f"[fallback] {dataset_name}: no 'test' split, using 'train'")
                dataset = load_dataset(dataset_name, split="train")
            else:
                raise
        # dataset = dataset.filter(lambda example: example['level'] == 'Level 5')
        print(f"Starting from index {args.resume_id} out of {len(dataset)} examples.")
        dataset = dataset.select(range(args.resume_id, len(dataset)))
        n = dataset_name.lower()
        if "olympiad" in n:
            prompt_key = "question"
            answer_key = "final_answer"
        elif "amo" in n:
            prompt_key = "prompt"
            answer_key = "answer"
        elif "aime" in n or "amc23" in n or "hmmt" in n:
            prompt_key = "problem" if "problem" in dataset.column_names else "prompt"
            answer_key = "answer"
        elif "math" in n:
            prompt_key = "problem" if "problem" in dataset.column_names else "prompt"
            answer_key = "solution" if "solution" in dataset.column_names else "answer"
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
        prompt_template = PROMPT_TEMPLATES[args.prompt_template]
        dataset = dataset.map(lambda x: prepare_data(x, prompt_key, prompt_template), num_proc=args.dataset_num_proc)

        output_file = dataset_name.split("/")[-1] + '-' + args.split + '-temp_' + str(args.temperature) + "-top_p_" + str(args.top_p) + "-top_k_" + str(args.top_k) + f'{args.comment}.jsonl'
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)

        # Auto-resume: detect partial results and resume from last valid point
        total_examples = len(dataset)  # before any slicing
        if args.resume_id == 0:
            auto_resume_id, should_skip = _auto_resume(output_path, args.num_generation, total_examples)
            if should_skip:
                print(f"Skipping {dataset_name} — results already complete.")
                continue
            if auto_resume_id > 0:
                args.resume_id = auto_resume_id
                dataset = dataset.select(range(args.resume_id, len(dataset)))
                print(f"Auto-resuming {dataset_name} from example {args.resume_id}/{total_examples}.")
        elif args.resume_id > 0 and not os.path.exists(output_path):
            print(f"Warning: --resume_id={args.resume_id} but output file doesn't exist. Starting fresh.")
            args.resume_id = 0

        # Create/append JSONL output
        with open(output_path, 'w' if args.resume_id == 0 else 'a') as f:
            for i in tqdm(range(0, len(dataset), args.batch_size)):
                batch = dataset[i:i + args.batch_size]
                inputs = batch["prompt"]
                answers = batch[answer_key]

                # Generate the answer
                outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=True)
                results = [[_.outputs[l].text for l in range(len(_.outputs))] for _ in outputs]
                finish_reasons = [[_.outputs[l].finish_reason for l in range(len(_.outputs))] for _ in outputs]
                assert len(results[0]) == args.num_generation, f"Number of generations is not equal to {args.num_generation}, got {len(results[0])}"

                # Prepare all outputs for batch tokenization
                flat_outputs = []
                output_mapping = []  # To map back to original indices
                
                for j in range(len(results)):
                    for k in range(args.num_generation):
                        flat_outputs.append(results[j][k])
                        output_mapping.append((j, k))

                # Process the results
                output_idx = 0
                for j, (inp, q, a, r, fr) in enumerate(zip(inputs, batch[prompt_key], answers, results, finish_reasons)):
                    for k in range(args.num_generation):
                        qa_pair = {
                            "prompt": inp,
                            "vanilla_response": r[k],
                            "question": q,
                            "answer": a,
                            "question_id": args.resume_id + i + j,
                            "generation_id": k,
                        }
                        qa_pair["response"] = r[k]
                        qa_pair["finish_reason"] = fr[k]
                        output_idx += 1
                        n2 = dataset_name.lower()
                        if "olympiad" in n2:
                            gold_answer = a[0] if isinstance(a, list) else a
                            pred_answer = extract_answer_math(qa_pair["response"])
                        elif "amo" in n2:
                            gold_answer = extract_answer_math(a) if "boxed" in str(a) else a
                            pred_answer = extract_answer_math(qa_pair["response"])
                        elif "aime" in n2 or "amc23" in n2 or "hmmt" in n2:
                            gold_answer = a
                            pred_answer = extract_answer_math(qa_pair["response"])
                        elif "math" in n2:
                            gold_answer = extract_answer_math(a)
                            pred_answer = extract_answer_math(qa_pair["response"])
                        else:
                            gold_answer = a
                            pred_answer = extract_answer_math(qa_pair["response"])
                        # qa_pair["label"] = pred_answer == gold_answer
                        qa_pair["label"] = math_equal(pred_answer, gold_answer, timeout=True)
                        qa_pair["gold_answer"] = gold_answer
                        qa_pair["pred_answer"] = pred_answer
                        f.write(json.dumps(qa_pair) + '\n')
                f.flush()


if __name__ == "__main__":
    main()
