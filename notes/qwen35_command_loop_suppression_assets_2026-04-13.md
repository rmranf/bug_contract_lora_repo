# Qwen3.5 Command Loop Suppression Assets

## 왜 추가했는가

최신 `MC-153355` 실행에서는 더 이상 바로 `window_not_foreground`로 죽지 않고,
실제로 월드 생성 후 in-game까지 진입한 뒤 `/gamemode creative`를 반복하는
`command-semantic loop`가 드러났다.

즉 현재 문제는 단순히 "잘못된 명령을 고르느냐" 뿐 아니라,
"이미 성공한 명령을 계속 다시 치느냐"도 포함한다.

## 이번에 추가한 것

- 생성 스크립트:
  - [build_qwen35_command_loop_suppression_assets.py](/c:/bugcraft/bugcraft/experiments/tools/build_qwen35_command_loop_suppression_assets.py)
- 신규 보강셋:
  - [command_loop_suppression_training_chat_v1.jsonl](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_command_semantic_2026-04-08/command_loop_suppression_training_chat_v1.jsonl)
- manifest:
  - [command_loop_suppression_manifest_v1.json](/c:/bugcraft/bugcraft/experiments/results/tuning/qwen35_command_semantic_2026-04-08/command_loop_suppression_manifest_v1.json)

## 무엇을 학습시키는가

이 보강셋은 다음 규칙을 학습시키기 위한 것이다.

`/gamemode creative`가 이미 성공했다면, 같은 명령을 반복하지 말고 다음 step의 command로 넘어간다.

예:
- `MC-153355`
  - 완료된 명령: `/gamemode creative`
  - 다음 정답 명령: `/tp @s -83 -12 0`

## 현재 생성 결과

| 항목 | 개수 |
|---|---:|
| 신규 loop suppression 샘플 | `3` |
| base canonical | `479` |
| base mixed goldx10 | `722` |
| canonical + loop | `482` |
| mixed goldx10 + loopx10 | `752` |

## 현재 포함된 다음-step family

- `tp`
- `setblock`

## 해석

이번 보강셋은 command syntax 자체를 고치는 것보다,
`completed command reissue`를 줄이기 위한 작은 recovery dataset이다.

즉:
- 기존 2차 command-semantic 학습: `올바른 명령을 선택하도록`
- 이번 loop suppression 보강: `이미 끝난 명령을 반복하지 않도록`

## 다음 추천

1. `command_semantic_training_chat_mixed_goldx10_with_loopx10_v1.jsonl`로 재학습
2. `MC-153355`와 같은 `/gamemode` family 사례를 다시 dev subset에서 확인
3. 같은 패턴이 보이면 `tp -> summon -> setblock` 전이도 추가 보강
