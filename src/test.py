import argparse
import json
import os


def build_prompt(problem):
    return f"Please reason step by step, and put your final answer within \\boxed{{}}.{problem}"


def split_list(items, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params):
    import jsonlines
    from tqdm import tqdm

    data = []
    with jsonlines.open(json_filename) as infile:
        print(f"Loading data from {json_filename}")
        for item in infile:
            problem_key = next((key for key in ["problem", "question", "input", "content"] if key in item), None)
            answer_key = next((key for key in ["answer", "target", "solution", "ground_truth"] if key in item), None)
            if not problem_key or not answer_key:
                continue
            problem = item[problem_key]
            data.append(
                {
                    "prompt_ori": build_prompt(problem),
                    "answer": item[answer_key],
                }
            )

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["prompt_ori"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in data
    ]

    output_dir = os.path.dirname(file_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    for batch in tqdm(split_list(texts, batch_size), desc=f"Processing {json_filename}"):
        outputs = llm.generate(batch, sampling_params)
        for output in outputs:
            results.append([candidate.text for candidate in output.outputs])

    for result, item in zip(results, data):
        item["output"] = result

    with open(file_name, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Data written to {file_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run batched vLLM inference for JSONL math data.")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--input_files", nargs="+", required=True, help="List of input JSONL files")
    parser.add_argument("--output_files", nargs="+", required=True, help="List of output JSONL files")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--n", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_tokens", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.input_files) != len(args.output_files):
        raise ValueError("The number of input and output files must match")

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("Missing inference dependency. Install requirements.txt first.") from exc

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.8,
        max_model_len=args.max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    for input_path, output_path in zip(args.input_files, args.output_files):
        process_data(input_path, output_path, llm, args.batch_size, tokenizer, sampling_params)


if __name__ == "__main__":
    main()
