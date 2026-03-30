---
name: rl-train
description: "DGX Spark에서 GRPO 학습 실행"
user_invocable: true
---

# GRPO 학습 실행

DGX Spark에서 GRPO 강화학습을 실행한다.

## 사용법
```
/rl-train                          # 기본: 200 steps
/rl-train steps=500                # 스텝 수 지정
/rl-train steps=500 round=2        # 라운드 지정 (이전 체크포인트에서 이어서)
```

## 실행 절차

1. venv 활성화: `source /home/kdb/Desktop/RL/.venv/bin/activate`
2. 이전 라운드 체크포인트 확인:
   - `outputs/round_{N-1}/final/` 존재하면 `--model` 인자로 전달
   - 없으면 기본 모델 `Qwen/Qwen3-0.6B` 사용
3. 학습 실행:
```bash
cd /home/kdb/Desktop/RL
source .venv/bin/activate
python scripts/train_grpo_pure.py \
  --model {model_path} \
  --max-steps {steps} \
  --batch-size 1 \
  --num-generations 2 \
  --output-dir outputs/round_{N}
```
4. 학습 완료 후:
   - AUDIT.log에 결과 기록
   - 핵심 metrics 요약 (loss, accuracy_reward, format_reward)
   - 모델 저장 위치 안내

## OOM 발생 시
- `--num-generations 1`로 줄이기
- `--batch-size 1` 확인
- `TORCHDYNAMO_DISABLE=1` 환경변수 확인
