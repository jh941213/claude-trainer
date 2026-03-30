# Claude Trainer

**Claude Code Sub-Agent 기반 LLM 강화학습 & 지식증류 파이프라인**

Claude Code의 서브에이전트를 활용하여 소형 LLM(Qwen 0.6B~)을 학습시키는 프레임워크.
기존 RL 파이프라인의 데이터 생성, Reward Model, Evaluator를 **Claude 서브에이전트**로 대체합니다.

> Tested on **NVIDIA DGX Spark** (GB10 Grace Blackwell, 128GB Unified Memory)

---

## Architecture

```
Claude Code (Orchestrator)
│
├── Agent: Data Crawler    → 도메인 데이터 수집/생성
├── Agent: Teacher         → CoT 풀이 생성 (지식증류)
├── Agent: Reward Judge    → LLM-as-Judge 실시간 채점
├── Agent: Evaluator       → 약점 분석 + 전략 수립
│
└── DGX Spark (Training)
    ├── GRPO (강화학습)
    ├── Knowledge Distillation (지식증류 SFT)
    └── Agent Reward Loop (Claude-in-the-Loop RL)
```

## 3 Training Methods

### 1. GRPO (Group Relative Policy Optimization)
DeepSeek-R1 방식의 강화학습. Critic 모델 없이, 그룹 내 상대 비교로 학습.

```bash
python scripts/train_grpo_pure.py --max-steps 200 --batch-size 2
```

```
매 Step:
  문제 → 모델이 N개 답 생성 → Reward 채점 → 잘한 답 강화, 못한 답 억제
```

### 2. Knowledge Distillation (지식증류)
Claude(Teacher)가 생성한 고품질 CoT 풀이를 Student 모델이 따라하도록 SFT.

```bash
# Step 1: Claude Teacher가 풀이 생성 (Claude Code에서 서브에이전트 스폰)
# Step 2: Student 모델 학습
python scripts/train_distill.py \
  --teacher-data data/teacher/round_1.jsonl \
  --max-steps 100
```

```
Claude(Opus) → CoT 풀이 생성 → Student 모델이 모방 학습 (SFT)
```

### 3. Agent Reward Loop (Claude-in-the-Loop RL)
**진짜 on-policy RL 루프**에 Claude 서브에이전트가 Reward Model로 참여.
학습 프로세스와 Claude Code가 IPC(파일)로 실시간 통신.

```bash
# Terminal: 학습 시작 (completions 생성 후 대기)
python scripts/train_grpo_agent_reward.py --max-steps 50

# Claude Code: /rl-reward-loop (서브에이전트가 매 step 채점)
```

```
┌─ DGX (Training) ──────────────┐     ┌─ Claude Code ──────────────┐
│                                │     │                            │
│ model.generate(prompt)         │     │                            │
│ → .ipc/completions.jsonl ─────────→  │ rl-reward-judge 스폰       │
│ ... 대기 ...                   │     │ 에이전트가 채점             │
│                                │  ←──│ → .ipc/rewards.jsonl       │
│ rewards 읽음 → gradient update │     │                            │
│ 다음 step...                   │     │ 다음 completions 대기      │
└────────────────────────────────┘     └────────────────────────────┘
```

---

## Skills (Claude Code)

| Skill | Description |
|-------|-------------|
| `/rl-pipeline` | 전체 파이프라인 자동 실행 (Data → Teacher → Train → Eval → Analyze) |
| `/rl-train` | DGX Spark에서 GRPO 학습 실행 |
| `/rl-eval` | 벤치마크 평가 (GSM8K, MATH-500) |
| `/rl-teacher` | Claude(Opus)로 Teacher CoT 생성 |
| `/rl-analyze` | 학습 결과 분석 + 다음 라운드 전략 |
| `/rl-reward-loop` | Agent Reward Loop 감시 + 채점 |

## Sub-Agents

| Agent | Model | Role |
|-------|-------|------|
| `rl-data-crawler` | Sonnet | 웹 검색으로 도메인 데이터 수집 |
| `rl-teacher` | Opus | 고품질 CoT 풀이 생성 (지식증류) |
| `rl-reward-judge` | Sonnet | LLM-as-Judge 채점 (correctness + reasoning + format) |
| `rl-evaluator` | Sonnet | 학습 결과 분석, CONTINUE/STOP/SCALE_UP 결정 |
| `rl-engineer` | Sonnet | DGX Spark 학습 실행 + 하이퍼파라미터 조정 |

---

## Project Structure

```
claude-trainer/
├── scripts/
│   ├── train_grpo_pure.py          # GRPO 강화학습 (rule-based reward)
│   ├── train_grpo_agent_reward.py  # GRPO + Claude Agent Reward (on-policy RL)
│   ├── train_distill.py            # 지식증류 SFT (Teacher → Student)
│   ├── generate_completions.py     # 학생 모델 답변 생성
│   ├── generate_teacher_data.py    # Teacher 데이터 변환
│   ├── evaluate.py                 # 벤치마크 평가
│   ├── judge_outputs.py            # 에러 분석 리포트
│   ├── download_datasets.py        # HuggingFace 데이터셋 다운로드
│   └── orchestrate.py              # 파이프라인 오케스트레이터
├── agents/                         # Claude Code 서브에이전트 정의
├── skills/                         # Claude Code 스킬 정의
├── docs/
│   └── DGX_SPARK_GUIDE.md
├── requirements.txt
└── run_pipeline.sh
```

## Quick Start

```bash
# 1. Setup
git clone https://github.com/jh941213/claude-trainer.git
cd claude-trainer
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_datasets.py --datasets gsm8k gsm8k_test math_500

# 3. Train (pick one)
python scripts/train_grpo_pure.py --max-steps 200          # GRPO
python scripts/train_distill.py --teacher-data data/teacher/round_1.jsonl  # KD
python scripts/train_grpo_agent_reward.py --max-steps 50   # Agent RL

# 4. Evaluate
python scripts/evaluate.py --checkpoint outputs/grpo_run/final --benchmarks gsm8k
```

---

## Claude as RL Infrastructure

| RL Component | Traditional | Claude Trainer |
|-------------|-------------|----------------|
| **Data** | 고정 데이터셋 | Claude가 생성/수집 |
| **Reward Model** | 별도 학습 필요 | Claude가 실시간 채점 |
| **Evaluator** | 룰 기반 | Claude가 심층 분석 |
| **Teacher** | 별도 대형 모델 | Claude(Opus)가 CoT 생성 |

## Domain Extension

PDF나 도메인 문서를 넣으면 Claude가 분석 → 데이터 생성 → 지식증류:

```
Your PDF/Docs → Claude 분석 → Q&A 데이터 생성 → train_distill.py → 도메인 특화 모델
```

의료, 법률, 금융 등 어떤 도메인이든 Claude가 읽을 수 있으면 학습 데이터로 변환 가능.

---

## Recommended: Use with Autoresearch

[**Karpathy's autoresearch**](https://github.com/karpathy/autoresearch)와 함께 사용하면 **자동 ML 연구 루프**를 구성할 수 있습니다.

```
┌─ autoresearch (Brain) ─────────────────────┐
│                                             │
│  논문/데이터 분석 → 가설 수립 → 실험 설계    │
│                                             │
└──────────────┬──────────────────────────────┘
               ▼
┌─ claude-trainer (Hands) ───────────────────┐
│                                             │
│  데이터 생성 → 학습 실행 → 평가 → 결과 반환  │
│                                             │
└──────────────┬──────────────────────────────┘
               ▼
┌─ autoresearch ─────────────────────────────┐
│                                             │
│  결과 분석 → 다음 실험 설계 → 반복           │
│                                             │
└─────────────────────────────────────────────┘
```

| Role | autoresearch | claude-trainer |
|------|-------------|----------------|
| **Research** | 논문 탐색, 가설 수립, 실험 설계 | - |
| **Data** | - | Claude 에이전트가 생성/수집 |
| **Training** | - | DGX Spark에서 GRPO/KD 실행 |
| **Evaluation** | 결과 해석, 논문 작성 | 벤치마크 평가, 에러 분석 |

autoresearch가 "어떤 실험을 할지" 결정하고, claude-trainer가 "실제로 실행"합니다.
연구 가설 → 학습 → 검증의 전체 사이클을 자동화할 수 있습니다.

---

## Tested Results

| Method | Model | Steps | Result | Speed |
|--------|-------|-------|--------|-------|
| GRPO | Qwen3-0.6B | 200 | accuracy_reward 0.5 | ~9.5s/step |
| Knowledge Distill | Qwen3-0.6B | 50 | loss 2.2→1.0, 생성 테스트 정답 | ~3.5it/s |
| Agent Reward Loop | Qwen3-0.6B | 5 | IPC 통신 성공, on-policy 확인 | ~30s/step |

> NVIDIA DGX Spark (GB10, 128GB Unified Memory)

## Requirements

- Python 3.10+
- CUDA GPU (tested on DGX Spark GB10)
- Claude Code (for sub-agent features)
- PyTorch 2.x + CUDA

## License

MIT
