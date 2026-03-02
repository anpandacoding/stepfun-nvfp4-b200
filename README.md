# StepFun Step-3.5-Flash NVFP4 — Inference Guide

## Model

**[apandacoding/Step-3.5-Flash-NVFP4](https://huggingface.co/apandacoding/Step-3.5-Flash-NVFP4)** — 196B-parameter MoE model (288 experts, top-8 routing) quantized to NVFP4. Checkpoint size: ~114 GB.

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 120 GB total (e.g. 2x80 GB) | 192 GB total (e.g. 2x96 GB) |
| GPU arch | Blackwell (sm_100+) for native FP4 | Blackwell |
| System RAM | 64 GB | 128 GB+ |
| Disk | 150 GB free | 250 GB free |
| CPU | 8+ cores | 16+ cores |

**Tested configurations:**
- 2x NVIDIA RTX PRO 6000 Blackwell (96 GB each) — works, ~40 GB headroom for KV cache
- 2x NVIDIA B200 (192 GB each) — works with large KV cache budget

**TP constraints:** Tensor parallel size must evenly divide 64 attention heads. Valid: 1, 2, 4, 8, 16, 32, 64.

## Setup

### 1. Install vLLM

```bash
pip install vllm
```

Requires vLLM v0.16+ with Blackwell support.

### 2. Patch vLLM (required until upstream merges StepFun NVFP4 support)

Apply three patches to the installed vLLM:

**Patch A — MoE quant algo resolution** (`vllm/model_executor/layers/quantization/modelopt.py`)

In `ModelOptMixedPrecisionConfig._resolve_quant_algo()`, after the prefix-based lookup (strategy 3), add:

```python
if prefix.endswith(".experts"):
    parent_dot = prefix.rsplit(".experts", 1)[0] + "."
    for key, info in self.quantized_layers.items():
        if key.startswith(parent_dot):
            return info["quant_algo"].upper()
```

Without this, MoE layers fall back to unquantized bf16 and OOM.

**Patch B — NVFP4 weight mappings** (`vllm/model_executor/models/step3p5.py`)

Replace `expert_params_mapping` with:

```python
expert_params_mapping = [
    (".moe.experts.w13_weight", ".moe.gate_proj.weight", "w1"),
    (".moe.experts.w13_weight", ".moe.up_proj.weight", "w3"),
    (".moe.experts.w2_weight", ".moe.down_proj.weight", "w2"),
    (".moe.experts.w13_weight_scale", ".moe.gate_proj.weight_scale", "w1"),
    (".moe.experts.w13_weight_scale", ".moe.up_proj.weight_scale", "w3"),
    (".moe.experts.w2_weight_scale", ".moe.down_proj.weight_scale", "w2"),
    (".moe.experts.w13_weight_scale_2", ".moe.gate_proj.weight_scale_2", "w1"),
    (".moe.experts.w13_weight_scale_2", ".moe.up_proj.weight_scale_2", "w3"),
    (".moe.experts.w2_weight_scale_2", ".moe.down_proj.weight_scale_2", "w2"),
    (".moe.experts.w13_input_scale", ".moe.gate_proj.input_scale", "w1"),
    (".moe.experts.w13_input_scale", ".moe.up_proj.input_scale", "w3"),
    (".moe.experts.w2_input_scale", ".moe.down_proj.input_scale", "w2"),
]
```

**Patch C — Scalar tensor handling** (`vllm/model_executor/models/step3p5.py`)

In `Step3p5Model.load_weights()`, replace the shape check block with:

```python
if loaded_weight.ndim == 0:
    loaded_weight = loaded_weight.unsqueeze(0).expand(moe_expert_num)
elif (
    loaded_weight.shape[0] == 1
    and loaded_weight.shape[0] != moe_expert_num
):
    loaded_weight = loaded_weight.expand(
        moe_expert_num, *loaded_weight.shape[1:]
    )
assert loaded_weight.shape[0] == moe_expert_num
```

### 3. Launch Server

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model apandacoding/Step-3.5-Flash-NVFP4 \
    --quantization modelopt \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
```

First launch downloads ~114 GB from HuggingFace (cached for subsequent runs).

## Inference

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "apandacoding/Step-3.5-Flash-NVFP4",
    "messages": [{"role": "user", "content": "Explain quantum computing briefly."}],
    "max_tokens": 256
  }'
```

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="apandacoding/Step-3.5-Flash-NVFP4",
    messages=[{"role": "user", "content": "Explain quantum computing briefly."}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

## Quantization Details

| Component | Format | Precision |
|-----------|--------|-----------|
| MoE expert MLPs (layers 3-44) | NVFP4 | 4-bit FP, block_size=16 |
| Dense MLPs (layers 0-2) | bf16 | Full precision |
| Attention (all layers) | bf16 | Full precision |
| Gate/router | bf16 | Full precision |
| Shared experts | bf16 | Full precision |
| Embeddings / lm_head | bf16 | Full precision |

Original bf16 model: ~392 GB. NVFP4 checkpoint: ~114 GB (3.4x compression).
