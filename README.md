qwen35 LoRA training bundle

This repo now contains:

1. Earlier contract-first assets in the repo root
2. Second-stage command-semantic assets under:
   - `experiments/tools/`
   - `experiments/results/tuning/qwen35_command_semantic_2026-04-08/`
   - `notes/`

Recommended server path:
- `/home/suho/bug_contract_lora_repo/`

Second-stage command-semantic training entrypoint:
- `experiments/tools/run_qwen35_command_semantic_lora_linux.sh`

Typical usage on the Linux server:

```bash
cd /home/suho/bug_contract_lora_repo
python3 -m venv .venv-command-semantic-lora
source .venv-command-semantic-lora/bin/activate
pip install -r experiments/tools/requirements.contract_lora.txt
CUDA_VISIBLE_DEVICES=1 bash experiments/tools/run_qwen35_command_semantic_lora_linux.sh
```
