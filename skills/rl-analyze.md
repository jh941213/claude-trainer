---
name: rl-analyze
description: "학습 결과 분석 + 다음 라운드 전략 수립 (rl-evaluator 에이전트 활용)"
user_invocable: true
---

# 학습 결과 분석

평가 결과를 분석하고 다음 라운드 전략을 수립한다. **rl-evaluator** 에이전트를 스폰하여 심층 분석.

## 사용법
```
/rl-analyze                        # 최신 라운드 분석
/rl-analyze round=2                # 특정 라운드 분석
```

## 실행 절차

1. 평가 결과 확인: `eval_results/round_{N}/report.json`
2. 기본 분석 실행:
```bash
cd /home/kdb/Desktop/RL
source .venv/bin/activate
python scripts/judge_outputs.py --eval-dir eval_results/round_{N}
```
3. 이전 라운드 비교 (있으면):
```bash
python scripts/judge_outputs.py \
  --eval-dir eval_results/round_{N} \
  --compare-with eval_results/round_{N-1}
```
4. **rl-evaluator** 에이전트 스폰하여 심층 분석:
   - 입력: report.json + analysis.md + details.jsonl
   - 분석: 토픽별 정확도, 에러 패턴, 난이도별 성능
   - 출력: 결정 (CONTINUE/STOP/SCALE_UP) + focus_topics + 전략 제안

## 결정 기준
| GSM8K 정확도 | 결정 | 다음 액션 |
|-------------|------|----------|
| > 85% | SCALE_UP | 더 어려운 데이터셋 또는 큰 모델 |
| 70-85% | CONTINUE | 약점 토픽 집중 학습 |
| 40-70% | CONTINUE | 스텝 수 증가 + 데이터 다양화 |
| < 40% | CONTINUE | 기초 산술/대수 집중 + curriculum learning |
