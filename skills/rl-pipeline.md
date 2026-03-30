---
name: rl-pipeline
description: "Claude 서브에이전트 기반 강화학습 파이프라인 - Claude가 문제 생성 + 채점 + 분석"
user_invocable: true
---

# Claude-in-the-Loop RL Pipeline

Claude 서브에이전트가 RL의 핵심 컴포넌트를 담당하는 파이프라인.
기존 정책(데이터셋+exact match)을 Claude로 대체.

## 사용법
```
/rl-pipeline                    # 1라운드 전체 실행
/rl-pipeline round=N            # N라운드부터
```

## 파이프라인 (매 라운드)

```
┌─────────────────────────────────────────────┐
│ Phase 1: Claude가 문제 생성/수집             │
│   rl-data-crawler 에이전트 스폰              │
│   → data/problems/round_N.jsonl             │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Phase 2: 학생 모델이 답 생성 (DGX Spark)     │
│   python scripts/generate_completions.py     │
│   → data/student_outputs/round_N.jsonl      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Phase 3: Claude가 채점 (LLM-as-Judge)       │
│   rl-reward-judge 에이전트 스폰              │
│   correctness + reasoning_quality + format   │
│   → data/rewards/round_N.jsonl              │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Phase 4: Claude reward로 GRPO 학습 (DGX)    │
│   python scripts/train_with_claude_rewards.py│
│   → outputs/round_N/final                   │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ Phase 5: Claude가 평가 + 다음 전략 결정      │
│   rl-evaluator 에이전트 스폰                 │
│   CONTINUE / STOP / SCALE_UP               │
│   약점 분석 → Phase 1에 피드백               │
└─────────────────────────────────────────────┘
```

## Phase별 실행 방법

### Phase 1: 문제 생성
**rl-data-crawler** 또는 **rl-teacher** 에이전트를 스폰:
- 라운드 1: GSM8K에서 50-100개 샘플링 + Claude가 난이도/토픽 분류
- 라운드 2+: 이전 라운드 약점 토픽 집중, Claude가 유사 문제 생성
- 출력: `data/problems/round_N.jsonl`

### Phase 2: Completions 생성
```bash
cd /home/kdb/Desktop/RL && source .venv/bin/activate
python scripts/generate_completions.py \
  --model {model_path} \
  --problems data/problems/round_N.jsonl \
  --num-samples 4 \
  --temperature 0.7 \
  --output data/student_outputs/round_N.jsonl
```

### Phase 3: Claude 채점
**rl-reward-judge** 에이전트를 스폰:
- 입력: `data/student_outputs/round_N.jsonl` (문제 + 학생 답변들)
- 채점 기준:
  - correctness (0 or 1): 정답 여부
  - reasoning_quality (0.0~1.0): 풀이 과정 논리성
  - format_compliance (0.0~0.3): 형식 준수
  - total: 합산 점수
- 출력: `data/rewards/round_N.jsonl`

### Phase 4: GRPO 학습
```bash
python scripts/train_with_claude_rewards.py \
  --model {model_path} \
  --rewards data/rewards/round_N.jsonl \
  --max-steps {steps} \
  --batch-size 2 \
  --num-generations 4 \
  --reward-mode combined \
  --output-dir outputs/round_N
```

### Phase 5: 평가 + 전략
**rl-evaluator** 에이전트 스폰:
- 입력: 학습 로그 + 채점 결과 통계
- 분석: 토픽별/난이도별 정확도, 에러 패턴
- 출력: 결정 + 다음 라운드 focus_topics
