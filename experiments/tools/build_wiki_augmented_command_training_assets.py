#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMMAND_TRAIN_DIR = ROOT / "experiments" / "results" / "tuning" / "qwen35_command_semantic_2026-04-08"
WIKI_DIR = ROOT / "experiments" / "results" / "tuning" / "wiki_command_reference_2026-04-13"


INPUT_OUTPUT_SPECS = [
    (
        COMMAND_TRAIN_DIR / "command_semantic_training_chat_with_loop_v1.jsonl",
        COMMAND_TRAIN_DIR / "command_semantic_training_chat_with_loop_with_wiki_v1.jsonl",
    ),
    (
        COMMAND_TRAIN_DIR / "command_semantic_training_chat_mixed_goldx10_with_loopx10_v1.jsonl",
        COMMAND_TRAIN_DIR / "command_semantic_training_chat_mixed_goldx10_with_loopx10_with_wiki_v1.jsonl",
    ),
    (
        COMMAND_TRAIN_DIR / "gamemode_focus_training_chat_v1.jsonl",
        COMMAND_TRAIN_DIR / "gamemode_focus_training_chat_with_wiki_v1.jsonl",
    ),
    (
        COMMAND_TRAIN_DIR / "gamemode_focus_training_chat_loopx20_v1.jsonl",
        COMMAND_TRAIN_DIR / "gamemode_focus_training_chat_loopx20_with_wiki_v1.jsonl",
    ),
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_examples(text: str) -> list[str]:
    seen: set[str] = set()
    commands: list[str] = []
    for match in re.findall(r"<code>([^<]+)</code>", text or ""):
        command = match.strip()
        if not command.startswith("/"):
            command = "/" + command
        if command in seen:
            continue
        seen.add(command)
        commands.append(command)
    return commands


def load_wiki_references() -> dict[str, dict]:
    raw = json.loads((WIKI_DIR / "wiki_command_reference_v1.json").read_text(encoding="utf-8"))
    references: dict[str, dict] = {}
    for family, payload in raw.items():
        references[family] = {
            "title": payload.get("title", ""),
            "syntax": compact_text(payload.get("syntax", "")),
            "arguments": compact_text(payload.get("arguments", "")),
            "result": compact_text(payload.get("result", "")),
            "canonical_examples": extract_examples(payload.get("examples", "")),
            "source": payload.get("page_path", ""),
        }
    return references


def augment_row(row: dict, wiki_refs: dict[str, dict]) -> tuple[dict, bool]:
    updated = copy.deepcopy(row)
    messages = updated.get("messages", [])
    if len(messages) < 2:
        return updated, False

    system_message = messages[0]
    user_message = messages[1]

    try:
        user_payload = json.loads(user_message["content"])
    except Exception:
        return updated, False

    family = user_payload.get("command_family")
    if not family or family not in wiki_refs:
        return updated, False

    if "wiki_reference" not in user_payload:
        user_payload["wiki_reference"] = wiki_refs[family]

    system_content = system_message.get("content", "")
    authority_notice = (
        " If `wiki_reference` is provided in the user payload, treat it as the authoritative "
        "syntax and example reference for that Minecraft command family."
    )
    if authority_notice.strip() not in system_content:
        system_message["content"] = system_content + authority_notice

    user_message["content"] = json.dumps(user_payload, ensure_ascii=False)

    metadata = updated.setdefault("metadata", {})
    metadata["wiki_augmented"] = True
    metadata["wiki_reference_family"] = family
    metadata["wiki_reference_source"] = wiki_refs[family].get("source", "")
    return updated, True


def main() -> None:
    wiki_refs = load_wiki_references()
    manifest: dict[str, object] = {
        "wiki_reference_path": str((WIKI_DIR / "wiki_command_reference_v1.json").relative_to(ROOT)),
        "families": sorted(wiki_refs.keys()),
        "outputs": {},
    }

    for input_path, output_path in INPUT_OUTPUT_SPECS:
        rows = read_jsonl(input_path)
        augmented_rows: list[dict] = []
        augmented_count = 0
        family_counts: dict[str, int] = {}

        for row in rows:
            updated, changed = augment_row(row, wiki_refs)
            augmented_rows.append(updated)
            if changed:
                augmented_count += 1
                family = updated.get("metadata", {}).get("wiki_reference_family", "unknown")
                family_counts[family] = family_counts.get(family, 0) + 1

        write_jsonl(output_path, augmented_rows)
        manifest["outputs"][output_path.name] = {
            "input": str(input_path.relative_to(ROOT)),
            "output": str(output_path.relative_to(ROOT)),
            "row_count": len(augmented_rows),
            "wiki_augmented_count": augmented_count,
            "family_counts": family_counts,
        }

    manifest_path = COMMAND_TRAIN_DIR / "wiki_augmented_training_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
