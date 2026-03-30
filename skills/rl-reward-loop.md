---
name: rl-reward-loop
description: "GRPO 학습 루프에서 서브에이전트가 실시간 reward 채점 (진��� on-policy RL)"
user_invocable: true
---

# Agent Reward Loop

GRPO 학습 프로세스와 IPC로 통신하며, 매 step마다 서브에이전트가 실시간 채점.

## 아키텍처

```
DGX Spark (Python)                    Claude Code
┌────────────────────┐                ┌──────────────────────┐
│ GRPO Training Loop │                │ /rl-reward-loop      │
│                    │                │                      │
│ Step N:            │                │                      │
│  model.generate()  │                │                      │
│  → completions.jsonl ──────────────→│ 파일 감지            │
│  ... 대기 ...      │                │ rl-reward-judge 스폰 │
│                    │                │ 에이전트가 채점       │
│                    │←────────────── │ → rewards.jsonl      │
│  rewards 읽음      │                │                      │
│  gradient update   ��                │                      │
│  Step N+1...       │                │ 다음 completions 대기│
└────────────────────┘                └──────────────────────┘
       .ipc/ 디렉토리로 통신
```

## 실행 방법

**터미널 1** (또는 background):
```bash
cd /home/kdb/Desktop/RL && source .venv/bin/activate
python scripts/train_grpo_agent_reward.py --max-steps 20
```

**Claude Code에서**:
```
/rl-reward-loop
```

## Claude Code 실행 절차

1. `.ipc/status.json` 읽어서 학습 상태 확인
2. 루프 시작:
   a. `.ipc/completions.jsonl` 파일 감시 (매 2초)
   b. 파일 발견 시 내용 읽기: step, prompt, completions, gold_answer
   c. **rl-reward-judge** 서브에이전트 스폰:
      - 입력: 문제 + 학생 답변들 + 정답
      - 채점: correctness(0/1) + reasoning_quality(0~1) + format(0~0.3)
      - 각 completion에 대해 total score 계산
   d. 채점 결과를 `.ipc/rewards.jsonl`에 작성:
      ```json
      {
        "step": 5,
        "rewards": [1.3, 0.0, 0.8, 0.2],
        "details": [
          {"correctness": 1.0, "reasoning": 0.8, "format": 0.3},
          ...
        ]
      }
      ```
   e. 다음 completions 대기
3. `.ipc/status.json`이 "done"이면 루프 종료

## 서브에이전트 채점 기준

rl-reward-judge에게 전달할 프롬프트:
- 문제 원문
- 정답
- 학생의 각 completion
- 채점 기준:
  - **correctness (0 or 1)**: 최종 답이 정답과 일치하는가
  - **reasoning_quality (0.0~1.0)**: 풀이 과정이 논리적인가, 중간 단계가 맞는가
  - **format_compliance (0.0~0.3)**: #### 사용, step 구분, 적절한 길이
  - **total**: correctness + reasoning_quality * 0.5 + format_compliance
