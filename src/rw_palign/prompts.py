from __future__ import annotations


def build_continuation_prompt(question: str, prefix: str) -> str:
    question_text = str(question).strip()
    prefix_text = str(prefix).strip()
    return (
        "Please continue from the draft and solve the problem step by step, "
        "and put your final answer within \\boxed{}.\n"
        f"Question: {question_text}\n"
        f"Prefix: {prefix_text}"
    )
