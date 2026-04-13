# qwen3.5 Gamemode-Focus Assets (2026-04-13)

## Why

Recent online reruns show that the most stubborn residual loop is still `/gamemode creative` retry behavior.
The with-loop adapter improved earlier fronts, but `MC-153355` still ends with repeated `/gamemode creative`
attempts and `Unknown command` feedback.

This focused asset pack isolates `gamemode`-family supervision so we can reinforce:

- valid `/gamemode creative` command generation
- recovery after repeated `gamemode` failures
- post-gamemode transition examples that should move to the next command instead of repeating

## Files

- [build_qwen35_gamemode_focus_assets.py](/c:/bugcraft/bugcraft/experiments/tools/build_qwen35_gamemode_focus_assets.py)
- [gamemode_focus_training_chat_v1.jsonl](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_command_semantic_2026-04-08/gamemode_focus_training_chat_v1.jsonl)
- [gamemode_focus_training_chat_loopx20_v1.jsonl](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_command_semantic_2026-04-08/gamemode_focus_training_chat_loopx20_v1.jsonl)
- [gamemode_focus_failure_queue_v1.jsonl](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_command_semantic_2026-04-08/gamemode_focus_failure_queue_v1.jsonl)
- [gamemode_focus_manifest_v1.json](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_command_semantic_2026-04-08/gamemode_focus_manifest_v1.json)
- [run_qwen35_gamemode_focus_lora_linux.sh](/c:/bugcraft/bugcraft/experiments/tools/run_qwen35_gamemode_focus_lora_linux.sh)

## Counts

- gamemode focus training rows: `229`
- gamemode focus training rows with loop oversample: `289`
- explicit loop-transition rows: `3`
- gamemode focus failure queue: `43`

## What To Check Next

1. Fine-tune with the focused dataset.
2. Rerun `MC-153355` first.
3. Check whether `/gamemode creative` is still retried after explicit failure feedback.
