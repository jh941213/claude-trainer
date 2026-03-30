# DGX Spark 학습 가이드

## Hardware Specs

| Spec | Value |
|------|-------|
| Chip | GB10 Grace Blackwell Superchip |
| Memory | 128GB unified LPDDR5x (273 GB/s) |
| GPU | 6,144 CUDA cores, 5th-gen Tensor Cores |
| Performance | 1 PFLOP FP4 sparse |
| CPU | 20-core ARM (10x Cortex-X925 + 10x A725) |
| CUDA Arch | sm_121 (Blackwell) |
| Size | 150mm x 150mm x 50mm, 1.2 kg |
| Power | 240W |

## Training Setup Options

### Option 1: Unsloth (추천 — RL 학습 지원)

```bash
# Docker
docker pull unsloth/unsloth:dgx-spark

# 또는 직접 설치
pip install unsloth trl transformers datasets peft accelerate
```

Unsloth은 DGX Spark에서 공식 지원하며, GRPO/DPO/PPO RL 학습을 메모리 효율적으로 실행.
OpenAI DevDay에서 gpt-oss-20b를 DGX Spark + Unsloth으로 RL 학습한 사례 있음.

### Option 2: NeMo AutoModel (NVIDIA 공식)

```bash
git clone https://github.com/NVIDIA/NeMo-Automodel
cd NeMo-Automodel
uv sync

# HuggingFace 토큰 설정
export HF_TOKEN=<your_token>

# 파인튜닝 실행
uv run --frozen --no-sync \
  examples/llm_finetune/finetune.py \
  -c examples/llm_finetune/config.yaml
```

NeMo AutoModel 특징:
- ARM64 + Blackwell 아키텍처 최적화
- FP8 precision 지원
- SFT, LoRA, QLoRA 지원
- HuggingFace 에코시스템 호환
- 1B-70B 모델 파인튜닝 가능

### Option 3: PyTorch Direct

```bash
pip install torch transformers trl peft accelerate
python scripts/train_grpo.py --model Qwen/Qwen3.5-0.8B
```

## Qwen3.5-0.8B on DGX Spark

### Memory Budget (128GB)

| Component | Memory (est.) |
|-----------|--------------|
| Model (bf16) | ~1.6GB |
| LoRA adapters | ~0.1GB |
| Optimizer states | ~3.2GB |
| GRPO: 8 generations batch | ~12.8GB |
| Gradients | ~1.6GB |
| Activations (batch=16) | ~8GB |
| **Total** | **~27GB** |
| **여유** | **~101GB** |

→ 메모리가 엄청 넉넉하므로 큰 배치, 긴 시퀀스 가능

### 성능 최적화 팁

1. **배치 크기 크게**: 메모리 여유 → batch_size=16~32
2. **bf16 사용**: Blackwell 텐서코어 활용
3. **Gradient checkpointing**: 굳이 필요 없음 (메모리 충분)
4. **num_generations 늘리기**: GRPO group size 8~16
5. **context length**: 2048~4096 토큰 가능

### llama.cpp 빌드 시 주의

```bash
# DGX Spark (GB10)은 sm_121 아키텍처
cmake -DCMAKE_CUDA_ARCHITECTURES="121" ..
```

## Troubleshooting

### OOM 발생 시 (가능성 낮음)
1. batch_size 줄이기
2. num_generations 줄이기 (8 → 4)
3. max_completion_length 줄이기

### 학습 속도 느릴 때
1. bf16 확인
2. DataLoader num_workers 늘리기
3. 불필요한 로깅 줄이기

### CUDA 관련
- DGX Spark CUDA arch: sm_121
- nvcc 버전 확인: `nvcc --version`
- GPU 상태: `nvidia-smi`
