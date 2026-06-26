import argparse
import json
import os
import signal
from pathlib import Path

from tqdm import tqdm

from rw_palign.verifier import local_verify_answer, verify_answer


try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None

try:
    from oat_math_grader import boxed_reward_fn as oat_evaluate
except ImportError:
    oat_evaluate = None


def timeout(seconds: int = 10):
    """
    A decorator to enforce timeouts on function execution on POSIX systems.
    """
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("Verification timed out.")

        def wrapper(*args, **kwargs):
            if os.name != "posix":
                return func(*args, **kwargs)
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator


@timeout(seconds=10)
def label_with_math_verify(preds: list[str], golden: str):
    """
    Perform symbolic verification when math_verify is installed.
    Falls back to exact/numeric local verification otherwise.
    """
    if parse is None or verify is None:
        labels = []
        parsed_outputs = []
        for prediction in preds:
            correct, parsed_answer, _ = verify_answer(
                prediction,
                golden,
                verifier=local_verify_answer,
            )
            labels.append(int(correct))
            parsed_outputs.append(parsed_answer or "")
        return labels, parsed_outputs

    parsed_preds = list(map(parse, preds))
    parsed_golden = list(map(parse, ["$" + golden + "$"] * len(preds)))
    try:
        labels = list(map(verify, parsed_golden, parsed_preds))
    except Exception:
        labels = [0] * len(preds)
    return [int(value) for value in labels], parsed_preds


def safe_oat_eval(pred: str, golden: str):
    """
    Use OAT evaluator for fallback grading when that optional dependency exists.
    """
    if oat_evaluate is None:
        return 0
    try:
        _, result = oat_evaluate(pred, golden, fast=False)
        return int(result == 1.0)
    except Exception:
        return 0


def evaluate_jsonl(input_file: str, output_file: str, use_oat: bool = True, any_true: bool = True):
    """
    Evaluate a JSONL dataset of math problems and write verified results.
    """
    results = []
    with open(input_file, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]

    print(f"Evaluating {len(data)} records from {input_file} ...")
    for item in tqdm(data):
        answer = str(item.get("answer", "")).strip()
        outputs = item.get("output")
        if isinstance(outputs, str):
            outputs = [outputs]
        if not outputs:
            continue

        try:
            labels, parsed_outputs = label_with_math_verify(outputs, answer)
        except Exception:
            labels = [0] * len(outputs)
            parsed_outputs = [""] * len(outputs)

        if use_oat:
            oat_labels = [safe_oat_eval(output, answer) for output in outputs]
            if any_true:
                labels = [int(label or oat_label) for label, oat_label in zip(labels, oat_labels)]
            else:
                labels = oat_labels

        item.update(
            {
                "label": labels,
                "passn": int(any(labels)),
                "output_ans": parsed_outputs,
            }
        )
        results.append(item)

    output_path = Path(output_file)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    metrics = _compute_metrics(results)
    print(f"pass@3: {metrics['pass_at_n']:.4f}")
    print(f"acc@3: {metrics['avg_answer_acc']:.4f}")
    print(f"pass@1=acc@1: {metrics['pass_at_1']:.4f}")
    return metrics


def _compute_metrics(results):
    total = len(results)
    pass_count = sum(1 for row in results if any(row.get("label", [])))
    all_answers_count = sum(len(row.get("label", [])) for row in results)
    all_answers_correct = sum(sum(row.get("label", [])) for row in results)
    first_answer_correct = sum(
        1 for row in results if row.get("label") and row["label"][0]
    )
    return {
        "total": total,
        "pass_count": pass_count,
        "pass_at_n": pass_count / total if total > 0 else 0.0,
        "avg_answer_acc": all_answers_correct / all_answers_count if all_answers_count > 0 else 0.0,
        "pass_at_1": first_answer_correct / total if total > 0 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated math JSONL outputs.")
    parser.add_argument("--input", required=True, help="Input JSONL path containing answer and output fields.")
    parser.add_argument("--output", required=True, help="Output JSONL path for labeled results.")
    parser.add_argument("--no_oat", action="store_true", help="Disable optional OAT fallback evaluator.")
    parser.add_argument(
        "--oat_only",
        action="store_true",
        help="Use OAT labels instead of OR-combining math_verify/local and OAT labels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_jsonl(
        input_file=args.input,
        output_file=args.output,
        use_oat=not args.no_oat,
        any_true=not args.oat_only,
    )


if __name__ == "__main__":
    main()
