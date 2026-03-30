"""
Claude Code 서브에이전트 Reward 기반 GRPO 학습
진짜 on-policy RL 루프: generate → 서브에이전트 채점 대기 → update

학습 루프가 completions를 파일에 쓰고,
Claude Code 서브에이전트가 채점 결과를 파일에 써주면 읽어서 학습.

Usage:
    python scripts/train_grpo_agent_reward.py --max-steps 50

    # Claude Code가 이 스크립트를 실행하면서 동시에
    # reward 파일을 감시 → 서브에이전트 스폰 → 채점 → 파일 작성
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer


# ═══════════════════════════════════════════════════════
# IPC 디렉토리: 학습 ↔ Claude Code 통신
# ═════════════════════���═════════════════════════════════

IPC_DIR = Path("/home/kdb/Desktop/RL/.ipc")
COMPLETIONS_FILE = IPC_DIR / "completions.jsonl"    # 학습 → Claude
REWARDS_FILE = IPC_DIR / "rewards.jsonl"             # Claude → 학습
STATUS_FILE = IPC_DIR / "status.json"                # 상태 공유
AUDIT_FILE = Path("/home/kdb/Desktop/RL/AUDIT.log")


def ipc_init():
    IPC_DIR.mkdir(exist_ok=True)
    # 이전 파일 정리
    for f in [COMPLETIONS_FILE, REWARDS_FILE]:
        if f.exists():
            f.unlink()


def ipc_write_status(step: int, phase: str, info: str = ""):
    STATUS_FILE.write_text(json.dumps({
        "step": step,
        "phase": phase,
        "info": info,
        "timestamp": datetime.now().isoformat(),
    }))


def ipc_write_completions(step: int, prompt: str, completions: list[str], gold_answer: str):
    """학습이 생성한 completions를 파일에 씀 → Claude가 읽을 것"""
    data = {
        "step": step,
        "prompt": prompt,
        "completions": completions,
        "gold_answer": gold_answer,
        "timestamp": datetime.now().isoformat(),
    }
    COMPLETIONS_FILE.write_text(json.dumps(data, ensure_ascii=False))
    ipc_write_status(step, "waiting_for_reward", f"{len(completions)} completions generated")


def ipc_wait_for_rewards(step: int, timeout: int = 300) -> list[float]:
    """Claude가 채점 결과를 써줄 때까지 대기"""
    start = time.time()
    while time.time() - start < timeout:
        if REWARDS_FILE.exists():
            try:
                data = json.loads(REWARDS_FILE.read_text())
                if data.get("step") == step:
                    rewards = data["rewards"]
                    # 읽었으면 파일 삭제 (다음 step을 위해)
                    REWARDS_FILE.unlink()
                    COMPLETIONS_FILE.unlink(missing_ok=True)
                    return rewards
            except (json.JSONDecodeError, KeyError):
                pass
        time.sleep(1)  # 1초마다 체크

    raise TimeoutError(f"Step {step}: Claude reward timeout ({timeout}s)")


def audit_log(msg: str):
    with open(AUDIT_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] train_agent_reward {msg}\n")


# ═══���═══════════════════════════════════════════════════
# Reward Function: Claude 서브에이전트 연동
# ═══════════════��═══════════════════════════════════════

_current_step = {"value": 0}
_current_prompt = {"value": ""}
_current_gold = {"value": ""}
_use_agent_reward = {"value": True}


def agent_reward(completions, answer, **kwargs):
    """
    Claude Code 서브에이전트가 채점하는 reward function.
    completions를 파일에 쓰고, 채점 결과가 올 때까지 대기.
    """
    step = _current_step["value"]

    # completion 텍스트 추출
    texts = []
    for c in completions:
        if isinstance(c, list):
            texts.append(c[-1].get("content", "") if c else "")
        else:
            texts.append(str(c))

    gold = answer[0] if answer else _current_gold["value"]

    if _use_agent_reward["value"]:
        # IPC: completions 쓰고 reward 대기
        prompt_text = _current_prompt["value"]
        ipc_write_completions(step, prompt_text, texts, str(gold))
        audit_log(f"Step {step}: wrote {len(texts)} completions, waiting for agent reward...")

        try:
            rewards = ipc_wait_for_rewards(step, timeout=300)
            audit_log(f"Step {step}: received agent rewards: {rewards}")
            return rewards
        except TimeoutError:
            audit_log(f"Step {step}: agent reward timeout, using fallback")
            # Fallback: exact match
            return _fallback_reward(texts, str(gold))
    else:
        return _fallback_reward(texts, str(gold))


def _fallback_reward(texts: list[str], gold: str) -> list[float]:
    """Fallback: 기본 exact match reward"""
    rewards = []
    for text in texts:
        pred = _normalize(_extract_answer(text))
        gold_norm = _normalize(gold)
        score = 1.0 if pred == gold_norm else 0.0
        if "####" in text:
            score += 0.2
        rewards.append(score)
    return rewards


def _extract_answer(text: str) -> str:
    h = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if h: return h.group(1).strip()
    b = re.findall(r"\\boxed\{([^}]+)\}", text)
    if b: return b[-1].strip()
    n = re.findall(r"-?\d+\.?\d*", text)
    if n: return n[-1]
    return text.strip()


def _normalize(answer: str) -> str:
    answer = answer.strip().replace(",", "").replace("$", "").replace("%", "")
    try: return str(float(answer))
    except ValueError: return answer.lower().strip()


# ═════════���════════════════════��════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = "You are a math reasoning assistant. Solve the problem step by step, then give your final answer after ####."


def load_gsm8k():
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


# ═════���═══════════════════════��═════════════════════════
# Main
# ═══════════════════════��═══════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", default="outputs/agent_rl_run")
    parser.add_argument("--no-agent", action="store_true", help="에이전트 없이 fallback만 사용")
    args = parser.parse_args()

    _use_agent_reward["value"] = not args.no_agent

    print(f"{'='*60}")
    print(f"Claude Agent Reward GRPO Training")
    print(f"  Model: {args.model}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Agent reward: {_use_agent_reward['value']}")
    print(f"  IPC dir: {IPC_DIR}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")

    # IPC 초기화
    ipc_init()
    ipc_write_status(0, "initializing", "Loading model...")

    # 모델 로드
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    print(f"Model: {model.num_parameters()/1e6:.0f}M params")

    # LoRA
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )

    # Dataset
    print("Loading GSM8K...")
    dataset = load_gsm8k()
    print(f"Dataset: {len(dataset)} samples")

    # GRPO Config
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

    # Trainer
    print("Initializing GRPO trainer with agent reward...")
    ipc_write_status(0, "ready", "Trainer initialized, starting training loop")
    audit_log("AGENT_RL_START")

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[agent_reward],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    print("\n" + "="*60)
    if _use_agent_reward["value"]:
        print("Starting GRPO with Claude Agent rewards...")
        print("  → 매 step마다 .ipc/completions.jsonl에 생성 결과를 씀")
        print("  → Claude Code가 서브에이전트로 채점 후 .ipc/rewards.jsonl에 씀")
        print("  → 학습이 reward를 읽고 업데이트")
    else:
        print("Starting GRPO with fallback rewards (no agent)...")
    print("="*60 + "\n")

    trainer.train()

    # Save
    print("\nSaving model...")
    ipc_write_status(args.max_steps, "saving", "Training complete")
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Saved to: {args.output_dir}/final")

    ipc_write_status(args.max_steps, "done", "All done")
    audit_log("AGENT_RL_DONE")
    print("Done!")


if __name__ == "__main__":
    main()
