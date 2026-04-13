# Qwen3.5 Wiki-Augmented Command Assets (2026-04-13)

## Summary

We added a wiki-grounded augmentation pass on top of the existing command-semantic training sets.

The idea is:

- runtime logs provide the bug-specific context and failure trajectory
- local Minecraft wiki pages provide the command-family syntax and canonical examples

This keeps the existing good labels while making the command-family reference more explicit during training.

## New asset builder

- `experiments/tools/build_wiki_augmented_command_training_assets.py`

## Local wiki source

- `experiments/results/tuning/wiki_command_reference_2026-04-13/wiki_command_reference_v1.json`

Families currently covered:

- `gamemode`
- `give`
- `setblock`
- `summon`
- `tp`

## Generated datasets

- `experiments/results/tuning/qwen35_command_semantic_2026-04-08/command_semantic_training_chat_with_loop_with_wiki_v1.jsonl`
- `experiments/results/tuning/qwen35_command_semantic_2026-04-08/command_semantic_training_chat_mixed_goldx10_with_loopx10_with_wiki_v1.jsonl`
- `experiments/results/tuning/qwen35_command_semantic_2026-04-08/gamemode_focus_training_chat_with_wiki_v1.jsonl`
- `experiments/results/tuning/qwen35_command_semantic_2026-04-08/gamemode_focus_training_chat_loopx20_with_wiki_v1.jsonl`

## Manifest

- `experiments/results/tuning/qwen35_command_semantic_2026-04-08/wiki_augmented_training_manifest_v1.json`

Current augmentation counts:

- general with-loop set: `350 / 482`
- general mixed set: `350 / 752`
- gamemode-focus set: `201 / 229`
- gamemode-focus loopx20 set: `201 / 289`

## New runner

- `experiments/tools/run_qwen35_gamemode_focus_wiki_lora_linux.sh`

This runner uses:

- phase1: `gamemode_focus_training_chat_with_wiki_v1.jsonl`
- phase2: `gamemode_focus_training_chat_loopx20_with_wiki_v1.jsonl`

## Why this matters

This is the first training variant where the command-family syntax reference is explicitly attached to the model input from a local wiki-derived source, instead of relying only on positive traces or human-corrected loop data.
