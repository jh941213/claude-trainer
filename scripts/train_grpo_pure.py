"""
Pure TRL GRPO Training on DGX Spark (no Unsloth)
Qwen3-0.6B 수학 추론 강화학습

Usage:
    python scripts/train_grpo_pure.py --max-steps 20
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import re
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


# ═══════════════════════════════════════════════════════
# Reward Functions
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a math reasoning assistant. Solve the problem step by step, then give your final answer after ####."""


def extract_answer(text: str) -> str:
    hash_match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if hash_match:
        return hash_match.group(1).strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
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


def accuracy_reward(completions, answer, **kwargs):
    rewards = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        pred = normalize(extract_answer(text))
        gold_norm = normalize(str(gold))
        rewards.append(1.0 if pred == gold_norm else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        else:
            text = str(completion)
        score = 0.0
        if "####" in text:
            score += 0.2
        if "step" in text.lower():
            score += 0.1
        rewards.append(score)
    return rewards


# ═══════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════

def load_gsm8k_for_grpo():
    ds = load_dataset("openai/gsm8k", "main", split="train")

    def transform(example):
        answer = example["answer"]
        final = answer.split("####")[-1].strip() if "####" in answer else answer
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": final,
        }

    return ds.map(transform, remove_columns=ds.column_names)


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="outputs/grpo_run")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Pure TRL GRPO Training on DGX Spark")
    print(f"  Model: {args.model}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    # 1. Load model + tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Model loaded: {model.num_parameters()/1e6:.0f}M params")

    # 2. LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. Dataset
    print("Loading GSM8K...")
    dataset = load_gsm8k_for_grpo()
    print(f"Dataset: {len(dataset)} samples")

    # 4. GRPO Config
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_completion_length=512,
        num_generations=args.num_generations,
        beta=0.04,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        max_grad_norm=1.0,
        report_to="none",
    )

    # 5. Trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[accuracy_reward, format_reward],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 6. Train
    print("\n" + "="*60)
    print("Starting GRPO training...")
    print("="*60 + "\n")
    trainer.train()

    # 7. Save
    print("\nSaving model...")
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Model saved to: {args.output_dir}/final")
    print("Done!")


if __name__ == "__main__":
    main()
