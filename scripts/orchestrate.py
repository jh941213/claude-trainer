"""
RL Pipeline Orchestrator
Claude Code 서브에이전트를 활용한 전체 학습 루프 관리

이 스크립트는 직접 실행하지 않고, /rl-pipeline 스킬에서 참조하는 설정/유틸리티를 제공.
실제 오케스트레이션은 Claude Code 서브에이전트가 수행.

Usage:
    # 직접 실행 (서브에이전트 없이, 규칙 기반만):
    python scripts/orchestrate.py --rounds 3

    # 권장: Claude Code에서 /rl-pipeline 사용
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
EVAL_DIR = PROJECT_DIR / "eval_results"
AUDIT_LOG = PROJECT_DIR / "AUDIT.log"

# 실제 동작하는 설정
DEFAULT_CONFIG = {
    "model": "Qwen/Qwen3-0.6B",
    "train_script": "scripts/train_grpo_pure.py",
    "eval_script": "scripts/evaluate.py",
    "default_steps": 200,
    "batch_size": 1,
    "num_generations": 2,
    "lr": 5e-6,
    "base_datasets": ["gsm8k", "gsm8k_test", "math_500"],
}


def audit_log(msg: str):
    """AUDIT.log에 이벤트 기록"""
    timestamp = datetime.now().isoformat()
    entry = f"[{timestamp}] orchestrator {msg}\n"
    print(f"[AUDIT] {msg}")
    with open(AUDIT_LOG, "a") as f:
        f.write(entry)


def run_command(cmd: list[str], description: str, timeout: int = 3600) -> tuple[bool, str]:
    """명령 실행 + 결과 반환"""
    audit_log(f"START {description}")
    cmd_str = " ".join(cmd)
    audit_log(f"  CMD: {cmd_str}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_DIR),
        )
        if result.returncode == 0:
            audit_log(f"DONE {description}")
            return True, result.stdout
        else:
            stderr = result.stderr[:500] if result.stderr else "No stderr"
            audit_log(f"FAIL {description}: {stderr}")
            return False, stderr
    except subprocess.TimeoutExpired:
        audit_log(f"TIMEOUT {description} (>{timeout}s)")
        return False, "Timeout"


def get_model_path(round_num: int) -> str:
    """이전 라운드 체크포인트 또는 기본 모델 경로"""
    if round_num > 1:
        prev = OUTPUTS_DIR / f"round_{round_num - 1}" / "final"
        if prev.exists():
            audit_log(f"Resuming from checkpoint: {prev}")
            return str(prev)
    return DEFAULT_CONFIG["model"]


def phase_data(round_num: int, focus_topics: list[str] | None = None) -> bool:
    """Phase 1: 데이터 확인/다운로드"""
    audit_log(f"PHASE_1_START Round {round_num}")

    gsm8k_path = DATA_DIR / "raw" / "gsm8k.jsonl"
    if not gsm8k_path.exists():
        ok, _ = run_command(
            [sys.executable, "scripts/download_datasets.py",
             "--datasets"] + DEFAULT_CONFIG["base_datasets"],
            "Download base datasets"
        )
        if not ok:
            return False

    if focus_topics:
        audit_log(f"Focus topics for data collection: {focus_topics}")
        # 서브에이전트 모드에서는 rl-data-crawler가 처리
        # 직접 실행 모드에서는 기존 데이터 사용

    audit_log(f"PHASE_1_DONE Round {round_num}")
    return True


def phase_train(round_num: int, max_steps: int = None) -> bool:
    """Phase 3: GRPO 학습"""
    audit_log(f"PHASE_3_START Round {round_num}")

    steps = max_steps or DEFAULT_CONFIG["default_steps"]
    model = get_model_path(round_num)
    output_dir = str(OUTPUTS_DIR / f"round_{round_num}")

    ok, output = run_command(
        [sys.executable, DEFAULT_CONFIG["train_script"],
         "--model", model,
         "--max-steps", str(steps),
         "--batch-size", str(DEFAULT_CONFIG["batch_size"]),
         "--num-generations", str(DEFAULT_CONFIG["num_generations"]),
         "--lr", str(DEFAULT_CONFIG["lr"]),
         "--output-dir", output_dir],
        f"GRPO Training Round {round_num} ({steps} steps)",
        timeout=7200,  # 2시간
    )

    audit_log(f"PHASE_3_{'DONE' if ok else 'FAIL'} Round {round_num}")
    return ok


def phase_evaluate(round_num: int) -> bool:
    """Phase 4: 벤치마크 평가"""
    audit_log(f"PHASE_4_START Round {round_num}")

    checkpoint = str(OUTPUTS_DIR / f"round_{round_num}" / "final")
    eval_output = str(EVAL_DIR / f"round_{round_num}")

    ok, output = run_command(
        [sys.executable, DEFAULT_CONFIG["eval_script"],
         "--checkpoint", checkpoint,
         "--benchmarks", "gsm8k", "math_500",
         "--output-dir", eval_output],
        f"Evaluation Round {round_num}",
        timeout=3600,
    )

    audit_log(f"PHASE_4_{'DONE' if ok else 'FAIL'} Round {round_num}")
    return ok


def phase_analyze(round_num: int) -> dict:
    """Phase 5: 결과 분석 + 다음 라운드 결정"""
    audit_log(f"PHASE_5_START Round {round_num}")

    report_file = EVAL_DIR / f"round_{round_num}" / "report.json"

    if not report_file.exists():
        audit_log("No report found - using default CONTINUE")
        return {"decision": "CONTINUE", "focus_topics": [], "gsm8k_accuracy": 0}

    with open(report_file) as f:
        report = json.load(f)

    benchmarks = report.get("benchmarks", {})
    gsm8k_acc = benchmarks.get("gsm8k", {}).get("accuracy", 0)
    math_acc = benchmarks.get("math_500", {}).get("accuracy", 0)

    audit_log(f"Results: GSM8K={gsm8k_acc:.1%}, MATH-500={math_acc:.1%}")

    # 이전 라운드와 비교
    if round_num > 1:
        prev_report = EVAL_DIR / f"round_{round_num - 1}" / "report.json"
        if prev_report.exists():
            with open(prev_report) as f:
                prev = json.load(f)
            prev_gsm8k = prev.get("benchmarks", {}).get("gsm8k", {}).get("accuracy", 0)
            delta = gsm8k_acc - prev_gsm8k
            audit_log(f"GSM8K delta: {delta:+.1%}")

            # 3라운드 연속 1% 미만 개선 → STOP
            if round_num >= 3 and abs(delta) < 0.01:
                return {"decision": "STOP", "reason": "Convergence (< 1% improvement)", "gsm8k_accuracy": gsm8k_acc}

    # 결정
    if gsm8k_acc > 0.85:
        result = {"decision": "SCALE_UP", "reason": f"GSM8K {gsm8k_acc:.1%} > 85%", "gsm8k_accuracy": gsm8k_acc}
    elif gsm8k_acc > 0.70:
        result = {"decision": "CONTINUE", "focus_topics": ["geometry", "combinatorics", "word_problems"], "gsm8k_accuracy": gsm8k_acc}
    elif gsm8k_acc > 0.40:
        result = {"decision": "CONTINUE", "focus_topics": ["arithmetic", "algebra", "multi_step"], "gsm8k_accuracy": gsm8k_acc}
    else:
        result = {"decision": "CONTINUE", "focus_topics": ["basic_arithmetic", "simple_algebra"], "gsm8k_accuracy": gsm8k_acc}

    # 분석 리포트 생성
    run_command(
        [sys.executable, "scripts/judge_outputs.py",
         "--eval-dir", str(EVAL_DIR / f"round_{round_num}")],
        "Generate analysis report",
    )

    audit_log(f"PHASE_5_DONE Round {round_num} decision={result['decision']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="RL Pipeline Orchestrator")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--skip-eval", action="store_true", help="학습만, 평가 건너뛰기")
    args = parser.parse_args()

    audit_log("=" * 60)
    audit_log(f"RL PIPELINE START rounds={args.rounds} steps={args.max_steps}")
    audit_log(f"  Model: {DEFAULT_CONFIG['model']}")
    audit_log("=" * 60)

    focus_topics = None

    for round_num in range(args.start_from, args.rounds + 1):
        audit_log(f"\n{'='*40} ROUND {round_num}/{args.rounds} {'='*40}")

        # Phase 1: Data
        if not phase_data(round_num, focus_topics):
            audit_log(f"Round {round_num} data prep failed")
            break

        # Phase 2: Teacher (서브에이전트 모드에서만 실행)
        audit_log(f"PHASE_2_SKIP Round {round_num} (direct mode - no teacher agent)")

        # Phase 3: Train
        if not phase_train(round_num, args.max_steps):
            audit_log(f"Round {round_num} training failed")
            break

        if args.skip_eval:
            continue

        # Phase 4: Evaluate
        if not phase_evaluate(round_num):
            audit_log(f"Round {round_num} evaluation failed (continuing anyway)")

        # Phase 5: Analyze
        analysis = phase_analyze(round_num)
        audit_log(f"Decision: {analysis['decision']}")

        if analysis["decision"] == "STOP":
            audit_log(f"Convergence reached: {analysis.get('reason', '')}")
            break
        elif analysis["decision"] == "SCALE_UP":
            audit_log(f"Ready for scale-up: {analysis.get('reason', '')}")
            break
        else:
            focus_topics = analysis.get("focus_topics", [])
            audit_log(f"Next round focus: {focus_topics}")

    audit_log("=" * 60)
    audit_log("RL PIPELINE COMPLETE")
    audit_log("=" * 60)


if __name__ == "__main__":
    main()
