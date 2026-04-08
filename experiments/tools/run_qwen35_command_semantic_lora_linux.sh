#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/suho/bug_contract_lora_repo}"
MODEL_PATH="${MODEL_PATH:-/home/suho/models/Qwen3.5-9B}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT_DIR/experiments/results/tuning/qwen35_command_semantic_2026-04-08}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-command-semantic-lora}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"

PHASE1_DATA="$TRAIN_DIR/command_semantic_training_chat_v1.jsonl"
PHASE2_DATA="$TRAIN_DIR/command_semantic_training_chat_mixed_goldx10_v1.jsonl"

PHASE1_OUT="$TRAIN_DIR/lora_runs/command_semantic_phase1_warmup"
PHASE2_OUT="$TRAIN_DIR/lora_runs/command_semantic_phase2_mixed"

echo "[info] ROOT_DIR=$ROOT_DIR"
echo "[info] MODEL_PATH=$MODEL_PATH"
echo "[info] TRAIN_DIR=$TRAIN_DIR"
echo "[info] CUDA_DEVICE=$CUDA_DEVICE"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[error] config.json not found under $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$PHASE1_DATA" ]]; then
  echo "[error] phase1 dataset not found: $PHASE1_DATA" >&2
  exit 1
fi

if [[ ! -f "$PHASE2_DATA" ]]; then
  echo "[error] phase2 dataset not found: $PHASE2_DATA" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/experiments/tools/requirements.contract_lora.txt"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

python "$ROOT_DIR/experiments/tools/train_qwen35_contract_lora.py" \
  --model-path "$MODEL_PATH" \
  --train-data "$PHASE1_DATA" \
  --output-dir "$PHASE1_OUT" \
  --num-train-epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --load-in-4bit \
  --gradient-checkpointing \
  --bf16

python "$ROOT_DIR/experiments/tools/train_qwen35_contract_lora.py" \
  --model-path "$MODEL_PATH" \
  --train-data "$PHASE2_DATA" \
  --output-dir "$PHASE2_OUT" \
  --num-train-epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --load-in-4bit \
  --gradient-checkpointing \
  --bf16

echo "[done] command-semantic LoRA warmup + mixed phase finished"
