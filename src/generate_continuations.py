from __future__ import annotations

import argparse
from pathlib import Path

from rw_palign.generation import (
    VLLMGenerator,
    build_generation_config,
    generate_continuation_samples,
    load_resume_keys,
    mock_generator,
)
from rw_palign.io import (
    ensure_output_path_available,
    load_continuation_samples,
    load_prefix_records,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate K continuations per prefix record.")
    parser.add_argument("--input", required=True, help="PrefixRecord JSONL input path.")
    parser.add_argument("--output", required=True, help="ContinuationSample JSONL output path.")
    parser.add_argument("--model", required=True, help="Model path or name.")
    parser.add_argument("--k", type=int, default=4, help="Samples per prefix.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top-p.")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum generated tokens.")
    parser.add_argument("--batch_size", type=int, default=128, help="Requested generation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--backend", choices=("vllm", "mock"), default="vllm", help="Generation backend.")
    parser.add_argument("--limit", type=int, help="Limit input records for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Skip existing record/sample outputs.")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed input rows.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.resume:
        ensure_output_path_available(output_path, overwrite=args.overwrite)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_generation_config(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        backend=args.backend,
    )
    if args.k <= 0:
        raise ValueError("k must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be non-negative")

    input_result = load_prefix_records(args.input, strict=args.strict)
    existing_rows = []
    resume_keys = set()
    if args.resume and output_path.exists():
        existing_result = load_continuation_samples(output_path)
        existing_rows = existing_result.records
        resume_keys = load_resume_keys(output_path)

    generator = mock_generator if args.backend == "mock" else VLLMGenerator(args.model)
    new_rows = generate_continuation_samples(
        input_result.records,
        generator=generator,
        generation_config=config,
        k=args.k,
        resume_keys=resume_keys,
        limit=args.limit,
    )

    write_jsonl(output_path, [*existing_rows, *new_rows])
    _write_resolved_config(output_path.parent, args, config)
    return 0


def _write_resolved_config(run_dir: Path, args: argparse.Namespace, config: dict[str, object]) -> None:
    config_path = run_dir / "config.resolved.json"
    if config_path.exists() and not args.overwrite:
        return
    write_json(
        config_path,
        {
            **config,
            "k": args.k,
            "input": str(Path(args.input)),
            "output": str(Path(args.output)),
            "limit": args.limit,
            "resume": args.resume,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
