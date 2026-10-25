#!/usr/bin/env python3
"""Build the labelled moderation dataset that backs the tests and the evaluator.

Templates (see :mod:`scripts.dataset_corpus`) rather than hand-typed rows, so the corpus
is reproducible (``--seed``), large enough to mean something, and explicit about *why*
each row carries its label - a precision/recall report is only debuggable if you can
read the failing row.

Per-row labels describe **policy outcomes**, not topics:

``stage1_expect``      what the toxicity/injection gate should decide (block/pass/flag)
``expect_status``      what the whole chain should return (allowed/rewritten/blocked)
``expect_categories``  at least one must appear in the response (empty = none expected)
``adversarial``        row exercises an evasion technique; reported separately

Scoring Stage 1 separately from the chain matters: a Stage-1 miss that Stage-2 happens
to rescue is still a Stage-1 bug, and an averaged accuracy number would hide it.

Usage:
    python scripts/generate_dataset.py --out tests/data/moderation_testset.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Scripts run both as ``python scripts/x.py`` and as ``scripts.x`` (the API imports the
# evaluator), so make the repository root importable before touching project modules.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dataset_corpus import (  # noqa: E402
    CATEGORY_BY_FAMILY,
    FAMILIES,
    expand_markers,
)


def build_rows(seed: int = 7) -> list[dict[str, Any]]:
    """Expand the corpus tuples into labelled dataset rows (order is deterministic)."""
    rows: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for family, entries in FAMILIES:
        category = CATEGORY_BY_FAMILY[family]
        for raw_text, expect_status, expect_categories, stage1_expect, rationale, adversarial in entries:
            text = expand_markers(raw_text)
            counters[family] = counters.get(family, 0) + 1
            rows.append(
                {
                    "id": f"{family}-{counters[family]:03d}",
                    "family": family,
                    "category": category,
                    "lang": _detect_lang(text),
                    "text": text,
                    "expect_status": expect_status,
                    "expect_categories": expect_categories,
                    "stage1_expect": stage1_expect,
                    "adversarial": adversarial,
                    "rationale": rationale,
                }
            )
    # A fixed seed shuffles the interleave order (so evaluation is not biased by family
    # ordering) while keeping row ids stable across regenerations.
    random.Random(seed).shuffle(rows)
    return rows


def _detect_lang(text: str) -> str:
    bengali = sum(1 for ch in text if 0x0980 <= ord(ch) <= 0x09FF)
    return "bn" if bengali >= 3 else "en"


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``.jsonl`` dataset; tolerates blank lines and comments."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="tests/data/moderation_testset.jsonl", help="destination .jsonl path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rows = build_rows(seed=args.seed)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[1] / out_path
    write_dataset(out_path, rows)

    if not args.quiet:
        summary: dict[str, dict[str, int]] = {}
        for row in rows:
            bucket = summary.setdefault(row["category"], {"total": 0, "blocked": 0, "rewritten": 0, "allowed": 0, "adversarial": 0})
            bucket["total"] += 1
            bucket[row["expect_status"]] = bucket.get(row["expect_status"], 0) + 1
            bucket["adversarial"] += 1 if row["adversarial"] else 0
        print(f"wrote {len(rows)} rows -> {out_path}")
        for category, counts in sorted(summary.items()):
            print(f"  {category:16} {counts['total']:3}  blocked={counts.get('blocked', 0):3} rewritten={counts.get('rewritten', 0):3} allowed={counts.get('allowed', 0):3} adversarial={counts.get('adversarial', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
