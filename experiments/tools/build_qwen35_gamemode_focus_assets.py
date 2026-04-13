from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_gamemode_training_row(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") or {}
    if metadata.get("command_family") == "gamemode":
        return True
    if metadata.get("completed_command") == "/gamemode creative":
        return True
    if metadata.get("next_command_family") == "gamemode":
        return True
    payload = json.dumps(row, ensure_ascii=False)
    return "/gamemode creative" in payload


def _is_gamemode_failure_row(row: dict[str, Any]) -> bool:
    if row.get("loop_dominant_invalid_command_family") == "gamemode":
        return True
    if row.get("top_command_family") == "gamemode":
        return True
    if "gamemode" in (row.get("step_command_families") or []):
        return True
    payload = json.dumps(row, ensure_ascii=False)
    return "/gamemode creative" in payload


def main() -> None:
    base_dir = Path("experiments/results/tuning/qwen35_command_semantic_2026-04-08")
    input_training_path = base_dir / "command_semantic_training_chat_with_loop_v1.jsonl"
    input_failure_path = base_dir / "command_semantic_failure_queue_v1.jsonl"
    output_training_path = base_dir / "gamemode_focus_training_chat_v1.jsonl"
    output_training_loopx_path = base_dir / "gamemode_focus_training_chat_loopx20_v1.jsonl"
    output_failure_path = base_dir / "gamemode_focus_failure_queue_v1.jsonl"
    output_manifest_path = base_dir / "gamemode_focus_manifest_v1.json"
    loop_oversample = 20

    training_rows = _read_jsonl(input_training_path)
    failure_rows = _read_jsonl(input_failure_path)

    focus_training_rows = [row for row in training_rows if _is_gamemode_training_row(row)]
    focus_failure_rows = [row for row in failure_rows if _is_gamemode_failure_row(row)]
    loop_rows = [
        row
        for row in focus_training_rows
        if row.get("source_dataset") == "command_loop_suppression_v1"
    ]
    upsampled_training_rows = focus_training_rows + (loop_rows * loop_oversample)

    split_counts = Counter(str(row.get("split") or "unknown") for row in focus_training_rows)
    source_counts = Counter(
        str(row.get("source_dataset") or "unknown") for row in focus_training_rows
    )
    failure_split_counts = Counter(
        str(row.get("split") or "unknown") for row in focus_failure_rows
    )
    failure_code_counts = Counter(
        str(row.get("failure_code") or "unknown") for row in focus_failure_rows
    )

    _write_jsonl(output_training_path, focus_training_rows)
    _write_jsonl(output_training_loopx_path, upsampled_training_rows)
    _write_jsonl(output_failure_path, focus_failure_rows)
    _write_json(
        output_manifest_path,
        {
            "phase": "gamemode_focus_v1",
            "source_training_path": str(input_training_path).replace("\\", "/"),
            "source_failure_path": str(input_failure_path).replace("\\", "/"),
            "training_count": len(focus_training_rows),
            "training_loopx20_count": len(upsampled_training_rows),
            "loop_row_count": len(loop_rows),
            "loop_oversample": loop_oversample,
            "failure_count": len(focus_failure_rows),
            "split_counts": dict(sorted(split_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "failure_split_counts": dict(sorted(failure_split_counts.items())),
            "failure_code_counts": dict(sorted(failure_code_counts.items())),
            "output_training_path": str(output_training_path).replace("\\", "/"),
            "output_training_loopx_path": str(output_training_loopx_path).replace("\\", "/"),
            "output_failure_path": str(output_failure_path).replace("\\", "/"),
        },
    )


if __name__ == "__main__":
    main()
