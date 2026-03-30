"""
학생 모델로 completions 생성
Claude Judge가 채점할 데이터를 만드는 스크립트

Usage:
    python scripts/generate_completions.py --model Qwen/Qwen3-0.6B --problems data/problems/round_1.jsonl --num-samples 4
    python scripts/generate_completions.py --model outputs/round_1/final --problems data/problems/round_1.jsonl
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


SYSTEM_PROMPT = "You are a math reasoning assistant. Solve the problem step by step, then give your final answer after ####."


def generate_completions(
    model,
    tokenizer,
    problems: list[dict],
    num_samples: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    """각 문제에 대해 num_samples개의 completion 생성"""
    results = []

    for prob in tqdm(problems, desc="Generating completions"):
        question = prob.get("question") or prob.get("prompt", "")
        if isinstance(question, list):
            # 이미 메시지 형식이면 마지막 user 메시지 추출
            question = question[-1].get("content", "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        completions = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            completions.append(response)

        results.append({
            "question": question,
            "gold_answer": prob.get("gold_answer") or prob.get("answer", ""),
            "completions": completions,
            "metadata": {
                "source": prob.get("source", "unknown"),
                "difficulty": prob.get("difficulty", "unknown"),
                "topic": prob.get("topic", "unknown"),
            },
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--problems", required=True, help="문제 JSONL 경로")
    parser.add_argument("--output", help="출력 경로 (기본: data/student_outputs/round_N.jsonl)")
    parser.add_argument("--num-samples", type=int, default=4, help="문제당 생성할 답 수")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # 출력 경로
    output_path = args.output or f"data/student_outputs/completions.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 문제 로드
    problems = []
    with open(args.problems) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"Loaded {len(problems)} problems from {args.problems}")

    # 모델 로드
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {model.num_parameters()/1e6:.0f}M params")

    # 생성
    results = generate_completions(
        model, tokenizer, problems,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # 저장
    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_completions = sum(len(r["completions"]) for r in results)
    print(f"\nGenerated {total_completions} completions for {len(results)} problems")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
