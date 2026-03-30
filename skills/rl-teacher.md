---
name: rl-teacher
description: "Claude(Opus)로 수학 문제 CoT 풀이 생성 (지식증류용)"
user_invocable: true
---

# Teacher CoT 생성

Claude(Opus)를 Teacher로 활용하여 수학 문제에 대한 고품질 Chain-of-Thought 풀이를 생성한다.
생성된 데이터는 GRPO 학습의 보조 데이터 또는 SFT 지식증류에 사용.

## 사용법
```
/rl-teacher                        # GSM8K에서 50개 샘플 CoT 생성
/rl-teacher samples=100            # 100개 샘플
/rl-teacher round=2                # 라운드 2 데이터 생성
/rl-teacher topic=geometry         # 특정 토픽 집중
```

## 실행 절차

1. 입력 데이터 로드: `data/raw/gsm8k.jsonl`에서 샘플 선택
2. **rl-teacher** 에이전트 스폰 (model: opus):
   - 각 문제에 대해 상세 step-by-step 풀이 생성
   - 형식: `Step 1: ... Step 2: ... #### {answer}`
   - 다양한 풀이 전략 사용 (직접 계산, 방정식, 역추적 등)
3. 출력을 `data/teacher/round_{N}.jsonl`에 저장
4. GRPO 형식으로 변환:
```bash
python scripts/generate_teacher_data.py \
  --input data/teacher/round_{N}.jsonl \
  --output data/formatted/teacher_round_{N}_grpo.jsonl
```

## Teacher 에이전트 프롬프트

에이전트에게 전달할 내용:
- 문제 리스트 (question + gold_answer)
- 출력 형식 요구사항
- 풀이 품질 기준 (200-800 토큰, 모든 중간 단계 포함)
- 저장 경로

## 출력 스키마
```json
{
  "question": "원래 문제",
  "gold_answer": "정답",
  "teacher_solution": "Step 1: ... Step 2: ... #### 42",
  "strategy": "direct_calculation|equation|backtracking",
  "difficulty_assessed": "easy|medium|hard"
}
```
