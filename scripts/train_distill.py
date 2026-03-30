"""
Knowledge Distillation: Claude(Teacher) → Qwen(Student) SFT 학습
Teacher가 생성한 CoT 풀이를 Student 모델이 따라하도록 SFT

Usage:
    python scripts/train_distill.py --teacher-data data/teacher/round_1.jsonl --max-steps 50
    python scripts/train_distill.py --teacher-data data/teacher/round_1.jsonl --max-steps 200 --model outputs/round_1/final
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset


SYSTEM_PROMPT = "You are a math reasoning assistant. Solve the problem step by step, then give your final answer after ####."


# ═══════════════════════════════════════════════════════
# Dataset: Teacher CoT → SFT 형식
# ═══════════════════════════════════════════════════════

def load_teacher_dataset(path: str, tokenizer) -> Dataset:
    """Teacher CoT 데이터를 SFT 학습용 토큰으로 변환"""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items.append(item)

    print(f"Loaded {len(items)} teacher samples")

    # 메시지 형식으로 변환
    processed = []
    for item in items:
        question = item.get("question", "")
        solution = item.get("teacher_solution", "")
        gold = item.get("gold_answer", "")

        if not solution:
            continue

        # 풀이에 #### 답이 없으면 추가
        if "####" not in solution and gold:
            solution = solution.rstrip() + f"\n#### {gold}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]

        # 토큰화
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding=False,
        )

        # labels: user/system 부분은 -100 (loss 안 매김), assistant 부분만 학습
        # 간단히: 전체를 labels로 사용 (teacher forcing)
        tokenized["labels"] = tokenized["input_ids"].copy()
        processed.append(tokenized)

    return Dataset.from_list(processed)


# ═══════════════════════════════════════════════════════
# Data Collator
# ═══════════════════════════════════════════════════════

class SFTDataCollator:
    """Dynamic padding + labels masking"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-data", required=True, help="Teacher CoT JSONL path")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="outputs/distill_run")
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Knowledge Distillation: Teacher → Student SFT")
    print(f"  Student model: {args.model}")
    print(f"  Teacher data: {args.teacher_data}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    # 1. 모델 + 토크나이저 로드
    print("Loading student model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model: {model.num_parameters()/1e6:.0f}M params")

    # 2. LoRA 적용
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Teacher 데이터 로드
    print(f"\nLoading teacher data from {args.teacher_data}...")
    dataset = load_teacher_dataset(args.teacher_data, tokenizer)
    print(f"Dataset: {len(dataset)} samples")

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        warmup_steps=5,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
    )

    # 5. Trainer
    print("Initializing SFT trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTDataCollator(tokenizer),
    )

    # 6. Train
    print("\n" + "="*60)
    print("Starting Knowledge Distillation SFT...")
    print(f"  Student learns to mimic Teacher's CoT solutions")
    print("="*60 + "\n")
    trainer.train()

    # 7. Save
    print("\nSaving distilled model...")
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Saved to: {args.output_dir}/final")

    # 8. Quick test: 생성 테스트
    print("\n" + "="*60)
    print("Quick generation test...")
    print("="*60)
    model.eval()

    test_q = "Sarah has 5 apples. She buys 3 more and gives 2 away. How many apples does she have?"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_q},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {test_q}")
    print(f"A: {response}")
    print("\nDone!")


if __name__ == "__main__":
    main()
