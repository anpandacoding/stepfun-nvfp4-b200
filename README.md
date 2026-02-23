# StepFun-3.5-Flash NVFP4 Quantization on 3x NVIDIA B200

End-to-end pipeline for quantizing [stepfun-ai/Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) (196B-parameter MoE, bf16) to NVFP4 using NVIDIA ModelOpt post-training quantization, running on 3x NVIDIA B200 GPUs.

## Hardware

- **GPUs**: 3x NVIDIA B200 (178 GB VRAM each, 535 GB total)
- **System RAM**: 2,267 GB
- **Disk**: 750 GB overlay (root), plus network-mounted `/workspace`

## Files

| File | Purpose |
|------|---------|
| `run_stepfun_nvfp4.py` | Main pipeline: download, quantize, export, and optionally serve via vLLM |
| `repack_moe_fp4.py` | Post-processing fix: repacks MoE expert weights from bf16 to NVFP4 |
| `stepfun-nvfp4-ckpt/` | Final NVFP4 checkpoint (112 GB, 126 safetensors shards) |

## Steps Completed

### 1. Install Dependencies

```bash
pip install -U nvidia-modelopt[hf] torch transformers accelerate datasets psutil
pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129
```

### 2. Run the Quantization Pipeline

```bash
python run_stepfun_nvfp4.py --export-dir ./stepfun-nvfp4-ckpt
```

This runs three stages:

**Stage 1 -- Model Loading (~18 min)**
- Downloads 44 checkpoint shards from HuggingFace
- Loads the 196B-param bf16 model across 3 GPUs via Accelerate `device_map="auto"`
- GPU distribution: 106 GB / 121 GB / 140 GB across GPU 0/1/2, zero CPU offload

**Stage 2 -- NVFP4 Post-Training Quantization (~5 min)**
- Collects 128 calibration samples from 7 diverse sources (CNN/DailyMail, WikiText, OpenWebText, GSM8K, OpenMathInstruct-2, HumanEval, Glaive function-calling) to maximize MoE expert activation coverage
- Patches ModelOpt's `MaxCalibrator.collect` for bf16 NaN/Inf safety
- Runs `mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)` -- 128 calibration forward passes
- Inserts 1344 TensorQuantizers across the model
- Saves intermediate quantized state (366.9 GB) for resume capability

**Stage 3 -- Export (~5 min)**
- Patches uncalibrated weight quantizers with weight-derived amax values
- Injects NVFP4 quantizers onto 126 MoELinear modules (the exporter doesn't recognize them natively)
- Calls `export_hf_checkpoint` to write 126 safetensors shards

### 3. Identify and Fix the MoE Export Bug

After export, inspection revealed a critical issue: ModelOpt's `export_hf_checkpoint` only packs **2D `nn.Linear` weights** to FP4. The **3D MoE expert weights** (`MoELinear`, shape `[288, 1280, 4096]`) were exported as raw **bfloat16**, making the checkpoint ~367 GB -- nearly the same as the original model.

**Diagnosis** (inspecting shard tensors):
- Dense/attention layers: correctly packed as `uint8` with fp8 scales
- MoE expert weights (up/gate/down_proj): still `bfloat16`, ~3 GB each, 126 total = ~378 GB

**Root cause**: `is_quantlinear()` in ModelOpt's export pipeline returns `False` for `MoELinear` modules, so `NVFP4QTensor.quantize()` is never called on them -- even though that function supports arbitrary tensor dimensions via `[...]` notation.

### 4. Repack MoE Weights to NVFP4

```bash
python repack_moe_fp4.py --ckpt-dir ./stepfun-nvfp4-ckpt
```

Post-processing script that:
1. Iterates all 126 safetensors shards
2. Identifies bf16 3D MoE weight tensors by name/dtype/shape
3. Quantizes each using ModelOpt's own `NVFP4QTensor.quantize()` with block_size=16
4. Replaces each bf16 weight with four tensors: `weight` (uint8 packed), `weight_scale` (fp8), `weight_scale_2` (fp32), `input_scale` (fp32)
5. Rewrites shards in-place and updates the safetensors index

**Result**: 131 seconds, checkpoint reduced from **367 GB to 112 GB** (3.3x compression).

### 5. Disk Space Management

The 750 GB overlay disk was a tight constraint:

| Stage | Disk Used | Notes |
|-------|-----------|-------|
| After export | 733 GB (98%) | 367 GB quant_state + 367 GB bf16-heavy export |
| Deleted quant_state | 530 GB (71%) | Freed 367 GB; quant_state only needed for resume |
| After MoE repack | 112 GB (15%) | Shards rewritten in-place, bf16 replaced with packed FP4 |

## Known Issues

### vLLM TP=3 Incompatibility

The model has 64 attention heads, which is not divisible by 3. vLLM requires `tensor_parallel_size` to evenly divide the head count. Valid TP sizes for this model: 1, 2, 4, 8, 16, 32, 64.

To serve on 2 of the 3 GPUs:
```bash
python run_stepfun_nvfp4.py --serve-only --export-dir ./stepfun-nvfp4-ckpt --tp-size 2
```

### NVFP4 On-Disk Format

NVFP4 weights are stored as `uint8` (not a dedicated FP4 dtype, which doesn't exist in PyTorch). Two FP4 E2M1 values are packed per byte:
- `weight`: `uint8 [..., N/2]` -- two 4-bit values per byte
- `weight_scale`: `float8_e4m3fn [..., N/16]` -- one scale per block of 16 elements
- `weight_scale_2`: `float32` scalar -- global double-quantization scale
- `input_scale`: `float32` scalar -- activation quantization scale

GPU kernels in vLLM/TensorRT-LLM unpack these at inference time.

## Checkpoint Summary

```
stepfun-nvfp4-ckpt/
  126 safetensors shards, 2337 tensor keys, 112 GB total
  config.json, hf_quant_config.json (quant_algo: NVFP4, group_size: 16)
  tokenizer.json, tokenizer_config.json, chat_template.jinja
```
