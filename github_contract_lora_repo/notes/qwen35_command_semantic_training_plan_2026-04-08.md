# qwen3.5 Command-Semantic 2차 학습 계획

## 왜 이 학습이 필요한가

1차 contract-first 학습은 `형식`을 많이 고쳤다.

- thought/action JSON 구조
- judgment / reflection 완결성
- parser가 읽을 수 있는 출력

하지만 지금 남은 큰 문제는 `명령 내용`이다.

예:
- `/give @s bundle{...}` 같이 마인크래프트 버전/문법에 안 맞는 명령
- `/gamemode creative`를 반복하다가 실제로는 진행이 안 되는 케이스
- `/setblock`, `/summon`, `/tp`가 상황에 맞지 않거나 실패 메시지를 무시하고 반복되는 케이스

즉 지금 단계는

`형식 학습 -> 완료`

다음으로

`명령 의미(command semantic) 학습 -> 시작`

으로 보는 게 맞다.

## 현재 준비된 자산

기준 디렉터리:
- [qwen35_command_semantic_2026-04-08](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_command_semantic_2026-04-08)

핵심 수치:

| 항목 | 개수 | 뜻 |
|---|---:|---|
| positive command examples | `452` | 이미 잘 동작했던 명령 예시 |
| failure queue | `70` | 잘못된 명령/루프 후보 |
| auto gold draft | `70` | 자동으로 만든 초안 정답 |
| training-ready gold | `2` | 지금 바로 학습에 넣어도 안전한 보수적 정답 |
| canonical training set | `454` | positive + 안전 gold |
| mixed goldx10 set | `472` | 안전 gold를 10배 가중한 혼합셋 |

주요 실패 family:

| command family | 건수 |
|---|---:|
| `gamemode` | `21` |
| `give` | `12` |
| `tp` | `7` |
| `summon` | `4` |
| `kill` | `2` |

주요 failure code:

| failure code | 건수 |
|---|---:|
| `iteration_limit_exceeded` | `55` |
| `thought_action_parse_failed` | `8` |
| `macro_bad_payload` | `7` |

## 형식 vs 내용

쉽게 구분하면:

| 구분 | 뜻 |
|---|---|
| 형식 | JSON 구조, 필드 이름, command_list 형태, parser가 읽을 수 있는지 |
| 내용 | 실제 Minecraft 명령이 맞는지, 버전에 맞는지, 상황에 맞는지 |

지금 남은 문제는 주로 `내용`이다.

## 2차 학습 목표

1. 실제로 실행 가능한 Minecraft 명령 비율 올리기
2. `Unknown or incomplete command` 같은 실패 줄이기
3. `/gamemode`, `/give`, `/tp`, `/summon`, `/setblock` 같은 자주 틀리는 family 안정화
4. 잘못된 명령 반복으로 이어지는 behavior loop 줄이기

## 이번 단계에서 바로 볼 지표

1. invalid command rate
2. command family별 실패 감소량
3. `macro_bad_payload` 감소 여부
4. `iteration_limit_exceeded` 중 명령 실패 기반 loop 감소 여부

## 실행 스크립트

- [train_qwen35_contract_lora.py](/c:/bugcraft/bugcraft/experiments/tools/train_qwen35_contract_lora.py)
- [run_qwen35_command_semantic_lora_linux.sh](/c:/bugcraft/bugcraft/experiments/tools/run_qwen35_command_semantic_lora_linux.sh)

## 권장 실행 순서

1. phase1 warmup
- dataset: `command_semantic_training_chat_v1.jsonl`

2. phase2 mixed
- dataset: `command_semantic_training_chat_mixed_goldx10_v1.jsonl`

3. offline command-semantic eval
- invalid command rate
- family별 command success

4. online dev subset
- 이전에 명령 오류가 많았던 bug 위주로 재검증

## 한 줄 결론

지금은 `형식`보다 `명령을 실제로 맞게 치는 법`을 더 학습시켜야 하는 단계다.
