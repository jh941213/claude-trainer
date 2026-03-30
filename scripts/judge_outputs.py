"""
학생 모델 출력 채점 도구
평가 결과를 분석하여 상세 리포트 생성

Usage:
    python scripts/judge_outputs.py --eval-dir eval_results/round_1
    python scripts/judge_outputs.py --eval-dir eval_results/round_1 --compare-with eval_results/round_0
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def analyze_errors(details: list[dict]) -> dict:
    """에러 패턴 분석"""
    errors = [d for d in details if not d.get("correct", False)]
    correct = [d for d in details if d.get("correct", False)]

    # 에러 유형 분류 (간단한 휴리스틱)
    error_types = Counter()
    for err in errors:
        pred = str(err.get("predicted", ""))
        gold = str(err.get("gold", ""))

        if not pred or pred == "None":
            error_types["no_answer"] += 1
        elif pred.replace(".", "").replace("-", "").isdigit() and gold.replace(".", "").replace("-", "").isdigit():
            try:
                p, g = float(pred), float(gold)
                if abs(p - g) < abs(g) * 0.1:
                    error_types["close_arithmetic"] += 1
                else:
                    error_types["wrong_calculation"] += 1
            except ValueError:
                error_types["parse_error"] += 1
        else:
            error_types["format_or_concept"] += 1

    return {
        "total": len(details),
        "correct": len(correct),
        "errors": len(errors),
        "accuracy": len(correct) / len(details) if details else 0,
        "error_breakdown": dict(error_types),
        "sample_errors": errors[:5],  # 처음 5개 에러 샘플
    }


def compare_rounds(current_dir: str, previous_dir: str) -> dict:
    """라운드 간 비교"""
    curr_report = Path(current_dir) / "report.json"
    prev_report = Path(previous_dir) / "report.json"

    if not curr_report.exists() or not prev_report.exists():
        return {"error": "Report files not found"}

    with open(curr_report) as f:
        curr = json.load(f)
    with open(prev_report) as f:
        prev = json.load(f)

    comparison = {}
    for benchmark in curr.get("benchmarks", {}):
        curr_acc = curr["benchmarks"][benchmark].get("accuracy", 0)
        prev_acc = prev.get("benchmarks", {}).get(benchmark, {}).get("accuracy", 0)
        comparison[benchmark] = {
            "current": curr_acc,
            "previous": prev_acc,
            "delta": curr_acc - prev_acc,
            "improved": curr_acc > prev_acc,
        }

    return comparison


def generate_analysis_report(eval_dir: str, compare_dir: str = None) -> str:
    """마크다운 분석 리포트 생성"""
    eval_path = Path(eval_dir)
    report_file = eval_path / "report.json"

    if not report_file.exists():
        return f"# Error\nNo report.json found in {eval_dir}"

    with open(report_file) as f:
        report = json.load(f)

    lines = [
        f"# Evaluation Analysis Report",
        f"",
        f"**Date**: {report.get('timestamp', 'N/A')}",
        f"**Model**: {report.get('model', 'N/A')}",
        f"",
        f"## Benchmark Results",
        f"",
        f"| Benchmark | Accuracy | Correct | Total |",
        f"|-----------|----------|---------|-------|",
    ]

    for bench, data in report.get("benchmarks", {}).items():
        acc = data.get("accuracy", 0)
        correct = data.get("correct", 0)
        total = data.get("total", 0)
        lines.append(f"| {bench} | {acc:.1%} | {correct} | {total} |")

    # 라운드 비교
    if compare_dir:
        comparison = compare_rounds(eval_dir, compare_dir)
        if "error" not in comparison:
            lines.extend([
                f"",
                f"## Round Comparison",
                f"",
                f"| Benchmark | Previous | Current | Delta |",
                f"|-----------|----------|---------|-------|",
            ])
            for bench, data in comparison.items():
                delta_str = f"+{data['delta']:.1%}" if data['delta'] >= 0 else f"{data['delta']:.1%}"
                emoji = "+" if data['improved'] else "-"
                lines.append(f"| {bench} | {data['previous']:.1%} | {data['current']:.1%} | {delta_str} {emoji} |")

    # 에러 분석
    for bench in report.get("benchmarks", {}):
        details_file = eval_path / f"{bench}_details.jsonl"
        if details_file.exists():
            details = load_jsonl(str(details_file))
            analysis = analyze_errors(details)
            lines.extend([
                f"",
                f"## Error Analysis: {bench}",
                f"",
                f"- Errors: {analysis['errors']}/{analysis['total']}",
                f"- Error types: {json.dumps(analysis['error_breakdown'], indent=2)}",
            ])

    # 결정 제안
    benchmarks = report.get("benchmarks", {})
    gsm8k_acc = benchmarks.get("gsm8k", {}).get("accuracy", 0)
    lines.extend([
        f"",
        f"## Recommendation",
        f"",
    ])
    if gsm8k_acc > 0.85:
        lines.append(f"**SCALE_UP**: GSM8K {gsm8k_acc:.1%} > 85%. Ready for harder datasets or larger model.")
    elif gsm8k_acc > 0.70:
        lines.append(f"**CONTINUE**: GSM8K {gsm8k_acc:.1%}. Focus on weak topics.")
    elif gsm8k_acc > 0.40:
        lines.append(f"**CONTINUE**: GSM8K {gsm8k_acc:.1%}. Increase training steps and data diversity.")
    else:
        lines.append(f"**CONTINUE**: GSM8K {gsm8k_acc:.1%}. Low baseline - need more steps and potentially curriculum learning.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--compare-with", help="Previous round eval dir for comparison")
    parser.add_argument("--output", help="Output report path (default: {eval-dir}/analysis.md)")
    args = parser.parse_args()

    report = generate_analysis_report(args.eval_dir, args.compare_with)

    output_path = args.output or str(Path(args.eval_dir) / "analysis.md")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
