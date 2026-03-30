"""
Teacher CoT 데이터를 GRPO 학습 형식으로 변환

Usage:
    python scripts/generate_teacher_data.py --input data/teacher/round_1.jsonl --output data/formatted/teacher_round_1.jsonl
    python scripts/generate_teacher_data.py --input data/teacher/round_1.jsonl --merge-with data/formatted/gsm8k_grpo.jsonl
"""

import json
import argparse
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} items to {path}")


def teacher_to_grpo(teacher_data: list[dict]) -> list[dict]:
    """Teacher CoT 데이터를 GRPO prompt/answer 형식으로 변환"""
    grpo_data = []
    for item in teacher_data:
        prompt = item.get("prompt") or item.get("question", "")
        answer = item.get("answer") or item.get("gold_answer", "")
        teacher_solution = item.get("teacher_solution") or item.get("solution", "")

        # prompt가 이미 메시지 리스트면 그대로 사용
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": "You are a math reasoning assistant. Solve the problem step by step, then give your final answer after ####."},
                {"role": "user", "content": str(prompt)},
            ]

        entry = {
            "prompt": messages,
            "answer": str(answer),
        }

        # teacher solution이 있으면 참조용으로 포함
        if teacher_solution:
            entry["teacher_solution"] = teacher_solution

        grpo_data.append(entry)

    return grpo_data


def teacher_to_sft(teacher_data: list[dict]) -> list[dict]:
    """Teacher CoT 데이터를 SFT(지식증류) 형식으로 변환"""
    sft_data = []
    for item in teacher_data:
        prompt = item.get("prompt") or item.get("question", "")
        teacher_solution = item.get("teacher_solution") or item.get("solution", "")

        if not teacher_solution:
            continue

        if isinstance(prompt, list):
            messages = prompt + [{"role": "assistant", "content": teacher_solution}]
        else:
            messages = [
                {"role": "system", "content": "You are a math reasoning assistant. Solve the problem step by step, then give your final answer after ####."},
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": teacher_solution},
            ]

        sft_data.append({"messages": messages})

    return sft_data


def merge_datasets(base_path: str, new_data: list[dict]) -> list[dict]:
    """기존 데이터셋에 새 데이터 병합 (중복 제거)"""
    base = load_jsonl(base_path) if Path(base_path).exists() else []

    # 프롬프트 기반 중복 체크
    existing_prompts = set()
    for item in base:
        p = item.get("prompt", "")
        if isinstance(p, list):
            p = str(p[-1].get("content", ""))
        existing_prompts.add(p[:200])  # 앞 200자로 비교

    added = 0
    for item in new_data:
        p = item.get("prompt", "")
        if isinstance(p, list):
            p = str(p[-1].get("content", ""))
        key = p[:200]
        if key not in existing_prompts:
            base.append(item)
            existing_prompts.add(key)
            added += 1

    print(f"Merged: {added} new items added (total: {len(base)})")
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Teacher data JSONL path")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument("--format", choices=["grpo", "sft"], default="grpo")
    parser.add_argument("--merge-with", help="Merge with existing dataset")
    args = parser.parse_args()

    teacher_data = load_jsonl(args.input)
    print(f"Loaded {len(teacher_data)} teacher samples from {args.input}")

    if args.format == "grpo":
        converted = teacher_to_grpo(teacher_data)
    else:
        converted = teacher_to_sft(teacher_data)

    if args.merge_with:
        converted = merge_datasets(args.merge_with, converted)
        output_path = args.merge_with
    else:
        output_path = args.output or args.input.replace(".jsonl", f"_{args.format}.jsonl")

    save_jsonl(converted, output_path)


if __name__ == "__main__":
    main()
