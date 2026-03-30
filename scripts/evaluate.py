"""
수학 추론 평가 스크립트
학습된 모델의 수학 문제 풀이 능력을 벤치마크 대비 평가

Usage:
    python scripts/evaluate.py --checkpoint outputs/final
    python scripts/evaluate.py --checkpoint outputs/final --benchmarks gsm8k math_500
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


SYSTEM_PROMPT = """You are a math reasoning assistant. Think step by step inside <think>...</think> tags, then give your final answer in \\boxed{}."""


# ═══════════════════════════════════════════════════════
# Answer Extraction & Comparison
# ═══════════════════════════════════════════════════════

def extract_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    hash_match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if hash_match:
        return hash_match.group(1).strip()
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return text.strip()


def normalize(answer: str) -> str:
    answer = answer.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return str(float(answer))
    except ValueError:
        return answer.lower().strip()


def is_correct(predicted: str, gold: str) -> bool:
    return normalize(extract_answer(predicted)) == normalize(gold)


# ═══════════════════════════════════════════════════════
# Benchmark Loaders
# ═══════════════════════════════════════════════════════

def load_benchmark(name: str):
    """벤치마크 데이터셋 로드. (question, answer) 튜플 리스트 반환"""
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [
            (item["question"], item["answer"].split("####")[-1].strip())
            for item in ds
        ]
    elif name == "math_500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            (item["problem"], item.get("answer", ""))
            for item in ds
        ]
    elif name == "minerva":
        ds = load_dataset("math-ai/minerva-math", split="test")
        return [
            (item["problem"], item.get("answer", ""))
            for item in ds
        ]
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════

def evaluate_model(
    model,
    tokenizer,
    benchmark: list[tuple[str, str]],
    max_new_tokens: int = 2048,
    batch_size: int = 1,
    temperature: float = 0.0,
) -> dict:
    """모델을 벤치마크에서 평가"""
    correct = 0
    total = 0
    results = []

    for question, gold_answer in tqdm(benchmark, desc="Evaluating"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else None,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_correct = is_correct(response, gold_answer)

        if pred_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "model_response": response,
            "predicted_answer": extract_answer(response),
            "correct": pred_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate math reasoning model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--pass-at-k", type=int, default=1, help="Number of samples for pass@k")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    report = {
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    for bench_name in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")

        benchmark = load_benchmark(bench_name)
        print(f"Loaded {len(benchmark)} problems")

        result = evaluate_model(
            model, tokenizer, benchmark,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        report["benchmarks"][bench_name] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }

        print(f"\nResults: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")

        # Save detailed results
        detail_file = output_dir / f"{bench_name}_details.jsonl"
        with open(detail_file, "w", encoding="utf-8") as f:
            for r in result["results"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Error analysis
        errors = [r for r in result["results"] if not r["correct"]]
        if errors:
            print(f"\nSample errors ({min(3, len(errors))} of {len(errors)}):")
            for e in errors[:3]:
                print(f"  Q: {e['question'][:80]}...")
                print(f"  Gold: {e['gold_answer']}")
                print(f"  Pred: {e['predicted_answer']}")
                print()

    # Save report
    report_file = output_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for bench, res in report["benchmarks"].items():
        print(f"  {bench:15s}: {res['accuracy']:.1%} ({res['correct']}/{res['total']})")
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
