# Qwen3.5 Command-Semantic Gold Expansion

Date: 2026-04-08

## Summary

We expanded the automatic `command-semantic` gold promotion rules for top command families.

- Before: `training_ready_gold_count = 2`
- After: `training_ready_gold_count = 27`

This means the second-stage command-semantic tuning set is now materially stronger than the initial seed-only state.

## Current Manifest

- positive command examples: `452`
- failure queue: `70`
- gold draft count: `70`
- training-ready gold: `27`
- canonical training count: `479`
- mixed goldx10 count: `722`

## What Changed

We now automatically promote a failure case to training-ready gold when all of the following hold:

1. The failure belongs to a top command family:
   - `gamemode`
   - `give`
   - `tp`
   - `summon`
   - `setblock`
2. There is exactly one plausible family-matching command candidate across:
   - report commands
   - step commands
   - observed commands
3. The candidate looks safe for training:
   - starts with `/`
   - no obvious `HERE` / truncation marker
   - no multi-command concatenation
   - for `summon` / `setblock`, braces and quotes are balanced

## Training-Ready Gold by Command Family

- `gamemode`: `16`
- `give`: `4`
- `tp`: `4`
- `summon`: `2`
- `item`: `1`

## Remaining Review Queue by Command Family

- `unknown`: `13`
- `give`: `8`
- `gamemode`: `5`
- `tp`: `3`
- `summon`: `2`
- `execute`: `1`
- `fill`: `1`
- `function`: `1`
- `gamerule`: `1`
- `keybind`: `1`
- `kill`: `1`
- `setworldname`: `1`
- `time`: `1`
- `worldborder`: `1`
- `worldmenu`: `1`

## Interpretation

This update confirms that a large fraction of the remaining loop cases are safely convertible into command-semantic supervision without manual relabeling.

The strongest immediate signal is:

- `gamemode`, `give`, and `tp` can now be trained at meaningful scale
- `summon` has started to become trainable, but still needs more reviewed gold
- the next bottleneck is no longer "can we build any semantic gold?" but "how fast can we review the remaining long-tail and ambiguous cases?"

## Recommended Next Step

1. Run the second-stage `command-semantic` LoRA using the updated training set.
2. In parallel, review the remaining queue with this family priority:
   - `give`
   - `summon`
   - `tp`
   - `setblock`
   - then the long-tail families
3. Re-run the same `ours-lite` subset and check:
   - invalid command rate
   - family-specific invalid rate
   - command-semantic loop rate

## Key Takeaway

The project is no longer blocked on the absence of command-semantic gold.

We now have enough training-ready command-semantic supervision to justify a real second-stage LoRA run, while continuing to improve the remaining ambiguous families through targeted review.
