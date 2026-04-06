qwen35 contract-first LoRA bundle

Files:
- `run_qwen35_contract_lora_linux.sh`
- `train_qwen35_contract_lora.py`
- `requirements.contract_lora.txt`
- `phase1_train.jsonl`
- `phase2_train.jsonl`

Recommended server path:
- `/home/suho/bugcraft_contract_lora/`

Typical usage on the Linux server:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.contract_lora.txt
CUDA_VISIBLE_DEVICES=1 bash run_qwen35_contract_lora_linux.sh
```
