# qwen3.5 Contract-First LoRA Runbook

## 1차 목표

- `macro_bad_payload` 줄이기
- `thought_action_parse_failed` 줄이기
- `judgment_missing_classification` 줄이기
- `reflection_parse_failed` 줄이기
- executable action rate 올리기

주의:
- 이번 단계는 `behavior loop`를 직접 줄이는 단계가 아님
- semantic drift sentinel은 같이 봐야 함

## 학습 데이터

Phase 1 warmup:
- [execution_contract_training_chat_v1.jsonl](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_ours_lite_clean82_2026-04-06/execution_contract_training_chat_v1.jsonl)

Phase 2 mixed:
- [execution_contract_training_chat_mixed_goldx10_v1.jsonl](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_ours_lite_clean82_2026-04-06/execution_contract_training_chat_mixed_goldx10_v1.jsonl)

Manifest:
- [execution_contract_training_manifest_v1.json](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_ours_lite_clean82_2026-04-06/execution_contract_training_manifest_v1.json)

## 실행 스크립트

학습:
- [train_qwen35_contract_lora.py](/c:/bugcraft/bugcraft/experiments/tools/train_qwen35_contract_lora.py)
- [run_qwen35_contract_lora_linux.sh](/c:/bugcraft/bugcraft/experiments/tools/run_qwen35_contract_lora_linux.sh)
- [requirements.contract_lora.txt](/c:/bugcraft/bugcraft/experiments/tools/requirements.contract_lora.txt)

평가 집계:
- [eval_contract_predictions.py](/c:/bugcraft/bugcraft/experiments/tools/eval_contract_predictions.py)

## 현재 서버 기준 권장값

- 학습 모델 경로: `/home/suho/models/Qwen3.5-9B`
- 장치: `CUDA_VISIBLE_DEVICES=1`
- 방식: `QLoRA`
- 이유:
  - GPU0는 서빙 중
  - GPU1에서 LoRA/QLoRA로 분리하는 것이 가장 안전함

학습 전용 venv 필요:
- 현재 서빙 venv에는 `peft`, `bitsandbytes`가 없음
- 따라서 별도 venv에서 학습하는 것이 맞음

## 추천 순서

1. Phase 1 warmup LoRA
- dataset: `execution_contract_training_chat_v1.jsonl`

2. Phase 2 mixed LoRA
- dataset: `execution_contract_training_chat_mixed_goldx10_v1.jsonl`

3. offline contract eval
- raw parse success
- post-validation valid output
- semantic accept

4. online dev subset

5. dev 15 bug full

6. 설계 freeze

7. test 15 bug full

8. full rerun

## 예시 실행

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/tools/train_qwen35_contract_lora.py \
  --model-path /home/suho/models/Qwen3.5-9B \
  --train-data experiments/results/tuning/qwen35_ours_lite_clean82_2026-04-06/execution_contract_training_chat_v1.jsonl \
  --output-dir experiments/results/tuning/qwen35_ours_lite_clean82_2026-04-06/lora_runs/contract_phase1_warmup \
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
```

Phase 2:

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/tools/train_qwen35_contract_lora.py \
  --model-path /home/suho/models/Qwen3.5-9B \
  --train-data experiments/results/tuning/qwen35_ours_lite_clean82_2026-04-06/execution_contract_training_chat_mixed_goldx10_v1.jsonl \
  --output-dir experiments/results/tuning/qwen35_ours_lite_clean82_2026-04-06/lora_runs/contract_phase2_mixed \
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
```

원클릭 실행:

```bash
bash experiments/tools/run_qwen35_contract_lora_linux.sh
```

## 평가에서 꼭 볼 3개

1. raw parse success가 얼마나 올랐는가
2. post-validation 없이도 valid output이 늘었는가
3. semantic accept가 유지되는가

한 줄 요약:
- 지금은 full rerun보다 먼저, contract-first LoRA 1차를 작게 돌려서 `형식 준수`와 `의미 유지`를 같이 보는 단계다.
