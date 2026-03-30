"""
RL 학습용 데이터셋 다운로드 스크립트
유명한 수학 추론 데이터셋들을 HuggingFace에서 가져온다.
"""

from datasets import load_dataset
from pathlib import Path
import json
import argparse


DATA_DIR = Path(__file__).parent.parent / "data"

# ═══════════════════════════════════════════════════════
# Training Datasets
# ═══════════════════════════════════════════════════════

TRAIN_DATASETS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "description": "Grade School Math 8K - 초등~중등 수학 (7.5K)",
    },
    "math": {
        "path": "hendrycks/competition_math",
        "name": None,
        "split": "train",
        "description": "MATH - 경시대회 수준 수학 (12.5K)",
    },
    "numina_math": {
        "path": "AI-MO/NuminaMath-1.5",
        "name": None,
        "split": "train",
        "description": "NuminaMath 1.5 - 860K 경시 수학 문제+풀이",
    },
    "open_r1_math": {
        "path": "open-r1/OpenR1-Math-220k",
        "name": None,
        "split": "train",
        "description": "OpenR1-Math - DeepSeek-R1 재현용 수학 (~220K)",
    },
    "deepscaler": {
        "path": "agentica-org/DeepScaleR-Preview-Dataset",
        "name": None,
        "split": "train",
        "description": "DeepScaleR - GRPO 학습용 수학 (40K)",
    },
}

# ═══════════════════════════════════════════════════════
# Evaluation Datasets
# ═══════════════════════════════════════════════════════

EVAL_DATASETS = {
    "gsm8k_test": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "description": "GSM8K Test (1.3K)",
    },
    "math_500": {
        "path": "HuggingFaceH4/MATH-500",
        "name": None,
        "split": "test",
        "description": "MATH-500 벤치마크",
    },
    "minerva_math": {
        "path": "math-ai/minerva-math",
        "name": None,
        "split": "test",
        "description": "Minerva Math 벤치마크",
    },
}


def download_dataset(name: str, config: dict, output_dir: Path) -> int:
    """데이터셋 하나를 다운로드하고 JSONL로 저장"""
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"  Source: {config['path']}")
    print(f"  Description: {config['description']}")
    print(f"{'='*60}")

    try:
        ds = load_dataset(
            config["path"],
            name=config.get("name"),
            split=config["split"],
            trust_remote_code=True,
        )

        output_file = output_dir / f"{name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  Saved: {output_file} ({len(ds)} samples)")
        return len(ds)

    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Skipping {name}...")
        return 0


def format_for_grpo(input_file: Path, output_file: Path):
    """GRPO 학습 형식으로 변환: {prompt, answer}"""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            # GSM8K 형식
            if "question" in item and "answer" in item:
                # GSM8K answer에서 최종 숫자 추출
                answer = item["answer"]
                if "####" in answer:
                    final_answer = answer.split("####")[-1].strip()
                else:
                    final_answer = answer
                samples.append({
                    "prompt": item["question"],
                    "answer": final_answer,
                    "full_solution": answer,
                    "source": "gsm8k",
                })

            # MATH 형식
            elif "problem" in item and "solution" in item:
                samples.append({
                    "prompt": item["problem"],
                    "answer": item.get("answer", ""),
                    "full_solution": item["solution"],
                    "level": item.get("level", ""),
                    "type": item.get("type", ""),
                    "source": "math",
                })

    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Formatted: {output_file} ({len(samples)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Download RL training datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "gsm8k_test", "math_500"],
        help="Datasets to download (default: gsm8k + eval sets)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Download training datasets only",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Download evaluation datasets only",
    )
    args = parser.parse_args()

    # Create directories
    raw_dir = DATA_DIR / "raw"
    formatted_dir = DATA_DIR / "formatted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    formatted_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    if args.all:
        targets = {**TRAIN_DATASETS, **EVAL_DATASETS}
    elif args.train_only:
        targets = TRAIN_DATASETS
    elif args.eval_only:
        targets = EVAL_DATASETS
    else:
        all_datasets = {**TRAIN_DATASETS, **EVAL_DATASETS}
        targets = {k: v for k, v in all_datasets.items() if k in args.datasets}

    for name, config in targets.items():
        count = download_dataset(name, config, raw_dir)
        total += count

    # Format training data for GRPO
    print(f"\n{'='*60}")
    print("Formatting for GRPO training...")
    print(f"{'='*60}")

    for name in ["gsm8k", "math"]:
        raw_file = raw_dir / f"{name}.jsonl"
        if raw_file.exists():
            format_for_grpo(raw_file, formatted_dir / f"{name}_grpo.jsonl")

    print(f"\nDone! Total samples downloaded: {total}")
    print(f"Raw data: {raw_dir}")
    print(f"Formatted data: {formatted_dir}")


if __name__ == "__main__":
    main()
