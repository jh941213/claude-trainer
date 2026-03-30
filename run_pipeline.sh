#!/bin/bash
# RL Pipeline 직접 실행 (Claude Code 서브에이전트 없이)
# 서브에이전트 활용: Claude Code에서 /rl-pipeline 사용

set -e

cd /home/kdb/Desktop/RL
source .venv/bin/activate

ROUNDS=${1:-1}
STEPS=${2:-200}

echo "=========================================="
echo "RL Pipeline - Direct Mode"
echo "  Rounds: $ROUNDS"
echo "  Steps per round: $STEPS"
echo "=========================================="

python scripts/orchestrate.py \
  --rounds "$ROUNDS" \
  --max-steps "$STEPS"
