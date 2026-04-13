from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a Minecraft command-loop recovery assistant. "
    "Return exactly one JSON object with keys `thought` and `command_list`. "
    "Assume the previous command already succeeded if the user says so. "
    "Do not repeat a completed command. "
    "Return the next valid Minecraft command for the specified release version. "
    "Do not add markdown fences. Do not explain outside the JSON."
)

SAFE_FAMILY_ALLOWLIST = {
    "gamemode",
    "give",
    "tp",
    "summon",
    "setblock",
}

COMMAND_PATTERN = re.compile(r"(/[\w:.[\]@=,+\-~{}\"' ]+)")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _normalize_command(command: str) -> str:
    text = str(command or "").strip()
    text = text.strip("`").strip('"').strip("'").strip()
    text = text.rstrip(".,;")
    return text


def _command_family(command: str) -> str:
    text = _normalize_command(command)
    match = re.match(r"^/([A-Za-z_]+)", text)
    return match.group(1).lower() if match else ""


def _extract_commands(step_text: str) -> list[str]:
    commands: list[str] = []
    for match in COMMAND_PATTERN.finditer(step_text or ""):
        candidate = _normalize_command(match.group(1))
        if candidate.startswith("/") and candidate not in commands:
            commands.append(candidate)
    return commands


def _flatten_step_commands(context: dict[str, Any]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for cluster in context.get("steps") or []:
        title = str(cluster.get("title") or "").strip()
        for step_text in cluster.get("steps") or []:
            for command in _extract_commands(str(step_text)):
                flattened.append(
                    {
                        "cluster_title": title,
                        "step_text": str(step_text),
                        "command": command,
                        "family": _command_family(command),
                    }
                )
    return flattened


def _latest_raw_log_for_bug(log_dir: Path, bug_id: str) -> str:
    candidates = sorted(
        log_dir.glob(f"action_log_{bug_id}_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return ""
    return str(candidates[-1]).replace("\\", "/")


def build_rows(
    gold_rows: list[dict[str, Any]],
    debug_dir: Path,
    log_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_bug_ids: set[str] = set()
    for gold_row in gold_rows:
        if not gold_row.get("use_for_training"):
            continue
        bug_id = str(gold_row.get("bug_id") or "").strip()
        if not bug_id or bug_id in seen_bug_ids:
            continue
        target_commands = (
            gold_row.get("draft_target", {}).get("command_list") or []
        )
        if not target_commands:
            continue
        first_target = str(target_commands[0])
        if '/gamemode creative' not in first_target:
            continue

        context_path = debug_dir / bug_id / "context.json"
        if not context_path.exists():
            continue
        context = _read_json(context_path)
        step_commands = _flatten_step_commands(context)
        gamemode_index = next(
            (
                index
                for index, item in enumerate(step_commands)
                if item["command"] == "/gamemode creative"
            ),
            None,
        )
        if gamemode_index is None or gamemode_index + 1 >= len(step_commands):
            continue

        next_item = step_commands[gamemode_index + 1]
        next_command = next_item["command"]
        next_family = next_item["family"]
        if next_family not in SAFE_FAMILY_ALLOWLIST:
            continue
        if next_command == "/gamemode creative":
            continue

        row = {
            "sample_id": f"{bug_id}:post_gamemode_recovery",
            "source_dataset": "command_loop_suppression_v1",
            "source_quality": "derived_next_step_after_gamemode",
            "split": gold_row.get("split", "train"),
            "target_type": "thought_action",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "target_type": "thought_action",
                            "task": (
                                "The previous command already succeeded. "
                                "Do not repeat it. "
                                "Return the next exact thought_action JSON "
                                "object with one valid Minecraft command."
                            ),
                            "bug_title": context.get("title", ""),
                            "release_version": context.get("release_version", ""),
                            "completed_command": "/gamemode creative",
                            "current_cluster": next_item["cluster_title"],
                            "next_step_text": next_item["step_text"],
                            "next_command_family": next_family,
                            "allowed_schema": {
                                "thought": "string",
                                "command_list": [
                                    "command(\"/minecraft command ...\")"
                                ],
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": (
                                "The previous '/gamemode creative' command has "
                                "already succeeded, so repeating it would be a loop. "
                                f"The next required step is to execute {next_command} "
                                "to continue the reproduction procedure."
                            ),
                            "command_list": [f'command("{next_command}")'],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "target_json": {
                "thought": (
                    "The previous '/gamemode creative' command has already "
                    f"succeeded, so the correct next step is {next_command}."
                ),
                "command_list": [f'command("{next_command}")'],
            },
            "metadata": {
                "bug_id": bug_id,
                "bug_title": context.get("title", ""),
                "release_version": context.get("release_version", ""),
                "completed_command": "/gamemode creative",
                "next_command": next_command,
                "next_command_family": next_family,
                "source_sample_id": gold_row.get("sample_id", ""),
                "raw_log_path": _latest_raw_log_for_bug(log_dir, bug_id),
                "context_path": str(context_path).replace("\\", "/"),
                "cluster_title": next_item["cluster_title"],
                "derivation": "step_sequence_after_gamemode",
            },
        }
        rows.append(row)
        seen_bug_ids.add(bug_id)
    return rows


def main() -> None:
    base_dir = Path("experiments/results/tuning/qwen35_command_semantic_2026-04-08")
    debug_dir = Path("experiments/results/debug")
    log_dir = Path("logs")
    gold_path = base_dir / "command_semantic_gold_training_ready_v1.jsonl"
    output_path = base_dir / "command_loop_suppression_training_chat_v1.jsonl"
    manifest_path = base_dir / "command_loop_suppression_manifest_v1.json"
    canonical_base_path = base_dir / "command_semantic_training_chat_v1.jsonl"
    mixed_base_path = base_dir / "command_semantic_training_chat_mixed_goldx10_v1.jsonl"
    merged_canonical_path = base_dir / "command_semantic_training_chat_with_loop_v1.jsonl"
    merged_mixed_path = (
        base_dir / "command_semantic_training_chat_mixed_goldx10_with_loopx10_v1.jsonl"
    )
    loop_oversample = 10

    gold_rows = _read_jsonl(gold_path)
    rows = build_rows(gold_rows, debug_dir=debug_dir, log_dir=log_dir)
    canonical_base_rows = _read_jsonl(canonical_base_path)
    mixed_base_rows = _read_jsonl(mixed_base_path)
    merged_canonical_rows = canonical_base_rows + rows
    merged_mixed_rows = mixed_base_rows + (rows * loop_oversample)

    _write_jsonl(output_path, rows)
    _write_jsonl(merged_canonical_path, merged_canonical_rows)
    _write_jsonl(merged_mixed_path, merged_mixed_rows)
    _write_json(
        manifest_path,
        {
            "phase": "command_loop_suppression_v1",
            "sample_count": len(rows),
            "loop_oversample": loop_oversample,
            "families": sorted(
                {
                    row["metadata"]["next_command_family"]
                    for row in rows
                    if row.get("metadata", {}).get("next_command_family")
                }
            ),
            "source_gold_path": str(gold_path).replace("\\", "/"),
            "output_path": str(output_path).replace("\\", "/"),
            "base_canonical_count": len(canonical_base_rows),
            "base_mixed_count": len(mixed_base_rows),
            "merged_canonical_count": len(merged_canonical_rows),
            "merged_mixed_count": len(merged_mixed_rows),
            "merged_canonical_path": str(merged_canonical_path).replace("\\", "/"),
            "merged_mixed_path": str(merged_mixed_path).replace("\\", "/"),
        },
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path).replace("\\", "/"),
                "manifest_path": str(manifest_path).replace("\\", "/"),
                "sample_count": len(rows),
                "merged_canonical_path": str(merged_canonical_path).replace("\\", "/"),
                "merged_mixed_path": str(merged_mixed_path).replace("\\", "/"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
