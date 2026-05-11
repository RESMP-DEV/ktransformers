#!/usr/bin/env python3
"""Prepare JSONL prompts for DeepSeek V4 Flash INT8 calibration.

The default source is Bartowski iMatrix Calibration v3, using the semantic
chunked Hugging Face mirror. A larger Pile+Bartowski mix can be appended when
the 168 V3 chunks are not enough for stable activation coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

DEFAULT_BARTOWSKI_V3 = "lemon07r/bartowski-imatrix-v3-semantic"
PILE_BARTOWSKI_V3 = "lemon07r/pile-calibration-v3"


def _iter_dataset_text(dataset_id: str, split: str) -> Iterable[tuple[int, str]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("prepare_mr_gptq_calibration_prompts.py requires `pip install datasets`") from exc

    dataset = load_dataset(dataset_id, split=split)
    for idx, row in enumerate(dataset):
        text = str(row.get("text", "")).strip()
        if text:
            yield idx, text


def write_prompts(
    output: Path,
    *,
    bartowski_dataset: str,
    split: str,
    max_bartowski_samples: int | None,
    include_pile_v3: bool,
    max_pile_samples: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for idx, text in _iter_dataset_text(bartowski_dataset, split):
            if max_bartowski_samples is not None and idx >= max_bartowski_samples:
                break
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "domain_tag": "instruction-following",
                        "source_dataset": bartowski_dataset,
                        "source_index": idx,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            written += 1

        if include_pile_v3:
            for idx, text in _iter_dataset_text(PILE_BARTOWSKI_V3, split):
                if idx >= max_pile_samples:
                    break
                handle.write(
                    json.dumps(
                        {
                            "text": text,
                            "domain_tag": "instruction-following",
                            "source_dataset": PILE_BARTOWSKI_V3,
                            "source_index": idx,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSONL prompt file")
    parser.add_argument("--bartowski-dataset", default=DEFAULT_BARTOWSKI_V3)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-bartowski-samples", type=int)
    parser.add_argument(
        "--include-pile-v3",
        action="store_true",
        help="Append a capped slice of lemon07r/pile-calibration-v3 after Bartowski V3.",
    )
    parser.add_argument("--max-pile-samples", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = write_prompts(
        args.output,
        bartowski_dataset=args.bartowski_dataset,
        split=args.split,
        max_bartowski_samples=args.max_bartowski_samples,
        include_pile_v3=args.include_pile_v3,
        max_pile_samples=args.max_pile_samples,
    )
    print(f"Wrote {count} calibration prompts to {args.output}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
