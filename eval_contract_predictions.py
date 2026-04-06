from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate contract prediction metrics from an evaluation JSONL file."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        required=True,
        help=(
            "JSONL path with one row per sample. "
            "Expected fields: sample_id, target_type, raw_parse_success, post_validation_valid, "
            "semantic_accept, optional regen_count."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Where to write the aggregate metrics JSON.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else round(num / den, 6)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.predictions_path)
    by_target = defaultdict(list)
    regen_hist = Counter()
    for row in rows:
        by_target[row.get("target_type", "<missing>")].append(row)
        regen_hist[row.get("regen_count", 0)] += 1

    overall = {
        "sample_count": len(rows),
        "raw_parse_success_rate": ratio(
            sum(1 for row in rows if row.get("raw_parse_success")), len(rows)
        ),
        "post_validation_valid_rate": ratio(
            sum(1 for row in rows if row.get("post_validation_valid")), len(rows)
        ),
        "semantic_accept_rate": ratio(
            sum(1 for row in rows if row.get("semantic_accept")), len(rows)
        ),
    }

    per_target = {}
    for target_type, target_rows in sorted(by_target.items()):
        per_target[target_type] = {
            "sample_count": len(target_rows),
            "raw_parse_success_rate": ratio(
                sum(1 for row in target_rows if row.get("raw_parse_success")),
                len(target_rows),
            ),
            "post_validation_valid_rate": ratio(
                sum(1 for row in target_rows if row.get("post_validation_valid")),
                len(target_rows),
            ),
            "semantic_accept_rate": ratio(
                sum(1 for row in target_rows if row.get("semantic_accept")),
                len(target_rows),
            ),
        }

    payload = {
        "overall": overall,
        "per_target_type": per_target,
        "regen_histogram": dict(sorted(regen_hist.items())),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
