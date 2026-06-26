import argparse
import json
import os


def build_prompt(question, sufficient_reasoning):
    return (
        "Please continue from the draft and solve the problem step by step, "
        "and put your final answer within \\boxed{}. "
        "I will provide you with some prior knowledge as a draft to assist you in solving the question."
        f"*Question*:{question}\n"
        f"*Prefix*:{sufficient_reasoning}"
    )


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params):
    import jsonlines
    from tqdm import tqdm

    data = []
    with jsonlines.open(json_filename) as infile:
        for item in infile:
            question = item["question"]
            sufficient_reasoning = item["sufficient_reasoning"]
            data.append(
                {
                    "question": question,
                    "sufficient_reasoning": sufficient_reasoning,
                    "prompt": build_prompt(question, sufficient_reasoning),
                }
            )

    existing = set()
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    existing_item = json.loads(line)
                    existing.add(existing_item["question"])
                except Exception:
                    continue
        print(f"[Resume] Found {len(existing)} existing entries. Will skip them.")

    output_dir = os.path.dirname(file_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Total {len(data)} samples, batch_size={batch_size}, total_batches={total_batches}")
    with open(file_name, "a", encoding="utf-8") as file:
        for batch_idx in tqdm(range(total_batches), total=total_batches, desc="Generating"):
            start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch_data = [item for item in data[start:end] if item["question"] not in existing]
            if not batch_data:
                continue

            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for item in batch_data
            ]

            try:
                outputs = llm.generate(texts, sampling_params)
            except Exception as exc:
                print(f"[Error] Batch {batch_idx} generation failed: {exc}")
                continue

            for output, item in zip(outputs, batch_data):
                item["output"] = output.outputs[0].text
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

            file.flush()
            os.fsync(file.fileno())
            print(f"[Saved] Batch {batch_idx + 1}/{total_batches} ({len(batch_data)} items) written.")

    print("All data processed and saved successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prefix-alignment continuations with vLLM.")
    parser.add_argument("--model", required=True, help="Model path or name.")
    parser.add_argument("--input", required=True, help="Input prefix JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("Missing prefix-alignment dependency. Install requirements.txt first.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    process_data(args.input, args.output, llm, args.batch_size, tokenizer, sampling_params)


if __name__ == "__main__":
    main()
