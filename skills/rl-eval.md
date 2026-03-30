---
name: rl-eval
description: "학습된 모델의 벤치마크 평가 실행"
user_invocable: true
---

# 모델 평가

학습된 모델을 GSM8K, MATH-500 벤치마크로 평가한다.

## 사용법
```
/rl-eval                           # 최신 체크포인트 평가
/rl-eval round=2                   # 특정 라운드 체크포인트 평가
/rl-eval checkpoint=outputs/test_run/final  # 직접 경로 지정
```

## 실행 절차

1. 체크포인트 경로 확인
2. 평가 실행:
```bash
cd /home/kdb/Desktop/RL
source .venv/bin/activate
python scripts/evaluate.py \
  --checkpoint {checkpoint_path} \
  --benchmarks gsm8k math_500 \
  --output-dir eval_results/round_{N}
```
3. 결과 분석:
```bash
python scripts/judge_outputs.py --eval-dir eval_results/round_{N}
```
4. 이전 라운드와 비교 (있으면):
```bash
python scripts/judge_outputs.py \
  --eval-dir eval_results/round_{N} \
  --compare-with eval_results/round_{N-1}
```

## 출력
- `eval_results/round_{N}/report.json` - 벤치마크별 정확도
- `eval_results/round_{N}/analysis.md` - 상세 분석 리포트
- `eval_results/round_{N}/{benchmark}_details.jsonl` - 문제별 결과

## 평가 후
- **rl-evaluator** 에이전트를 스폰하여 심층 분석 요청 가능
- 에이전트가 약점 토픽, 에러 패턴, 다음 라운드 전략 제안
