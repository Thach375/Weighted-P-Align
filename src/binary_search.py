import argparse
import json
import os
import time

from tqdm import tqdm


def load_model(model_name):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing truncation dependency. Install requirements.txt first.") from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def chat(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def split_sentences(text):
    """
    Simple sentence-level splitter.
    """
    sentences = text.split(". ")
    return [
        sentence.strip() + "." if not sentence.endswith(".") else sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def reasoning_sufficiency_check(question, reasoning_part, model, tokenizer):
    """
    Check whether the partial reasoning is sufficient.
    """
    prompt = f"""
You are a reasoning evaluator.

You are given a partial reasoning prefix extracted from a longer chain-of-thought.
Your task is to judge whether this prefix already contains the essential logical structure and key transformations needed to complete the solution.

- Reply "[ENOUGH]" if the prefix establishes the core reasoning steps such that the remaining reasoning is straightforward or routine.
- Reply "[NOT_ENOUGH]" if any crucial reasoning step is still missing, making it difficult to reliably complete the solution.

Reply with exactly one token: [ENOUGH] or [NOT_ENOUGH].

Question:
{question}

Partial reasoning:
{reasoning_part}
"""
    try:
        response = chat(prompt, model, tokenizer)
        is_sufficient = "[ENOUGH]" in response or response.strip() == "ENOUGH"
        return response, is_sufficient
    except Exception as exc:
        print(f"Model error: {exc}")
        return f"ERROR: {exc}", False


def find_minimal_sufficient_prefix(question, sentences, model, tokenizer, sleep_sec=0.5):
    """
    Return the minimal sufficient reasoning prefix found by binary search.
    """
    total = len(sentences)
    if total == 0:
        return "", 0, False, ""

    left, right = 1, total
    best_idx = None
    best_response = ""

    print("\n" + "=" * 80)
    print("Starting binary search for shortest sufficient prefix")
    print(f"Total sentences: {total}")
    print(f"Initial search interval: [{left}, {right}]")
    print("=" * 80)

    step = 1
    while left <= right:
        mid = (left + right) // 2
        prefix_text = " ".join(sentences[:mid])

        print(f"\nRound {step}")
        print(f"Current interval: left={left}, right={right}")
        print(f"Checking prefix length: {mid}/{total}")

        started_at = time.time()
        response, is_sufficient = reasoning_sufficiency_check(
            question,
            prefix_text,
            model,
            tokenizer,
        )
        elapsed = time.time() - started_at

        print(f"Model output: {response}")
        print(f"Decision: {'ENOUGH' if is_sufficient else 'NOT_ENOUGH'}")
        print(f"Elapsed: {elapsed:.2f}s")

        if is_sufficient:
            best_idx = mid
            best_response = response
            right = mid - 1
        else:
            left = mid + 1

        step += 1
        time.sleep(sleep_sec)

    if best_idx is None:
        print("No sufficient prefix found; falling back to full reasoning.")
        return " ".join(sentences), total, False, ""

    print(f"Shortest sufficient prefix length: {best_idx}/{total}")
    print(f"Prefix ratio: {best_idx / total:.4f}")
    return " ".join(sentences[:best_idx]), best_idx, True, best_response


def process_jsonl(input_file, output_file, model, tokenizer, sleep_sec=0.5):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8"):
        pass

    with open(input_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for index, line in enumerate(tqdm(lines, desc="Processing records")):
        data = json.loads(line)
        question = data.get("question", "")
        full_reasoning = data.get("Long-CoT", "")

        if not question or not full_reasoning:
            continue

        sentences = split_sentences(full_reasoning)
        if not sentences:
            continue

        prefix_text, prefix_len, ok, eval_resp = find_minimal_sufficient_prefix(
            question,
            sentences,
            model,
            tokenizer,
            sleep_sec=sleep_sec,
        )

        result = {
            "id": data.get("id", index),
            "answer": data.get("answer", ""),
            "question": question,
            "sufficient_reasoning": prefix_text,
            "sufficient_sentences": prefix_len,
            "total_sentences": len(sentences),
            "prefix_ratio": prefix_len / len(sentences),
            "is_sufficient": ok,
            "evaluator_response": eval_resp,
        }

        with open(output_file, "a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Find minimal sufficient prefixes with binary search.")
    parser.add_argument("--model", required=True, help="Evaluator model path or name.")
    parser.add_argument("--input", required=True, help="Input JSONL path with question and Long-CoT fields.")
    parser.add_argument("--output", required=True, help="Output prefix-record JSONL path.")
    parser.add_argument("--sleep_sec", type=float, default=0.5, help="Delay between model calls.")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)
    process_jsonl(args.input, args.output, model, tokenizer, sleep_sec=args.sleep_sec)
    print("All records processed.")


if __name__ == "__main__":
    main()
