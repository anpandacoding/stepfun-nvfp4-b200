#!/usr/bin/env python3
"""
StepFun-3.5-Flash NVFP4 Quantization & Inference Pipeline
==========================================================

Downloads stepfun-ai/Step-3.5-Flash (bf16) from HuggingFace, quantizes it to
NVFP4 via NVIDIA ModelOpt PTQ on 3× NVIDIA B200 GPUs, exports the quantized
checkpoint, and optionally launches a vLLM inference server.

Hardware: 3× NVIDIA B200 (192 GB HBM3e each, ~576 GB total VRAM)

The 196B-param bf16 model (~392 GB) fits within the combined 576 GB VRAM.
Accelerate's device_map="auto" distributes layers across the 3 GPUs; any
residual overflow spills to CPU RAM via standard PCIe/NVLink host access.

Install dependencies first:
    pip install -U nvidia-modelopt[hf] torch transformers accelerate datasets
    pip install -U vllm --pre \
        --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
        --extra-index-url https://download.pytorch.org/whl/cu129

Usage examples:
    # Run full pipeline (download -> quantize -> export)
    python run_stepfun_nvfp4.py --export-dir ./stepfun-nvfp4-ckpt

    # Quantize and clean up BF16 cache afterwards to reclaim ~392 GB disk
    python run_stepfun_nvfp4.py --export-dir ./stepfun-nvfp4-ckpt --cleanup-cache

    # Only launch vLLM server from an already-exported checkpoint
    python run_stepfun_nvfp4.py --serve-only --export-dir ./stepfun-nvfp4-ckpt

    # Override HF cache location (point to your largest volume)
    HF_HOME=/mnt/nvme/hf_cache python run_stepfun_nvfp4.py --export-dir ./stepfun-nvfp4-ckpt
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "stepfun-ai/Step-3.5-Flash"
DEFAULT_EXPORT_DIR = "./stepfun-nvfp4-ckpt"
CALIB_SIZE = 128
CALIB_MAX_SEQ_LEN = 512
CALIB_BATCH_SIZE = 1
TP_SIZE = 3  # 3× B200 GPUs; set to 0 for auto-detect from torch.cuda.device_count()

# Multi-source calibration data.  Diverse domains maximise the chance that
# the MoE router activates every expert at least once during calibration.
# Each entry: (dataset_id, config_or_None, text_field, per-source cap).
# If a source fails to load we redistribute its quota to the remaining ones.
CALIB_SOURCES: list[tuple[str, str | None, str, int]] = [
    ("cnn_dailymail",            "3.0.0",               "article",    16),  # news prose
    ("wikitext",                 "wikitext-103-raw-v1",  "text",       16),  # encyclopedic
    ("Skylion007/openwebtext",   None,                   "text",       16),  # diverse web text
    ("openai/gsm8k",             "main",                 "question",   24),  # grade-school math reasoning
    ("nvidia/OpenMathInstruct-2", None,                  "problem",    24),  # math instruction (NVIDIA-curated)
    ("openai/openai_humaneval",  None,                   "prompt",     16),  # Python code completion — uses 'test' split
    ("glaiveai/glaive-function-calling-v2", None,        "system",     16),  # tool-calling / JSON schemas
]


def _resolve_tp_size(requested: int) -> int:
    """Return the effective tensor-parallel size.

    A value of 0 means 'auto-detect from available GPUs'.
    """
    if requested > 0:
        return requested
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No CUDA GPUs detected — cannot determine TP size")
    return n


# =========================================================================
# Helpers — MoE Quantizer Injection & Fixups
# =========================================================================

def _patch_tensorquantizer_calibrator_compat(model: torch.nn.Module | None = None) -> int:
    """Ensure every TensorQuantizer on *model* has a usable ``_calibrator``.

    Deep-scans all module ``__dict__`` values (including nested dicts/lists)
    so quantizers stored under non-standard attribute names are covered.
    Does NOT monkeypatch the TensorQuantizer class — only fixes instances.
    """
    try:
        from modelopt.torch.quantization.nn import TensorQuantizer
        from modelopt.torch.quantization.calib.max import MaxCalibrator
    except Exception:
        return 0

    if model is None:
        return 0

    def _ensure(q: TensorQuantizer) -> bool:
        if getattr(q, "_calibrator", None) is not None:
            return False
        try:
            q._calibrator = MaxCalibrator(axis=None, track_amax=False)
        except Exception:
            return False
        return True

    fixed = 0

    def _scan_obj(obj):
        nonlocal fixed
        if isinstance(obj, TensorQuantizer):
            if _ensure(obj):
                fixed += 1
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _scan_obj(v)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                _scan_obj(v)

    for m in model.modules():
        for v in m.__dict__.values():
            _scan_obj(v)

    if fixed:
        log.info("[calibrator-compat] Patched calibrators on %d TensorQuantizer(s).", fixed)
    return fixed


_NVFP4_TQ_CFG = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "axis": None,
    "enable": True,
}


def _make_nvfp4_quantizer(*, enable_quant: bool = True, enable_calib: bool = False):
    """Create an NVFP4-configured TensorQuantizer with a real MaxCalibrator."""
    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.quantization.calib.max import MaxCalibrator

    q = TensorQuantizer(_NVFP4_TQ_CFG)
    q._if_quant = enable_quant
    q._if_calib = enable_calib
    if not hasattr(q, "_unsigned"):
        q._unsigned = False

    if getattr(q, "_calibrator", None) is None:
        q._calibrator = MaxCalibrator(axis=None, track_amax=False)

    return q


def _inject_moe_linear_quantizers(model: torch.nn.Module) -> int:
    """Attach NVFP4 quantizers to MoELinear modules without replacing forward().

    MoELinear is not nn.Linear, so mtq.quantize() skips it. We attach properly
    configured TensorQuantizers with weight-derived amax so the export pipeline
    can pack the 3-D weights to FP4.

    Does NOT override forward — the original MoELinear.forward is preserved.
    """
    count = 0
    for name, module in model.named_modules():
        if "MoELinear" not in type(module).__name__:
            continue
        if hasattr(module, "weight_quantizer"):
            log.debug("[inject-moe] %s already has weight_quantizer — skipping.", name)
            continue

        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, torch.nn.Parameter):
            continue

        dev = weight.device

        wq = _make_nvfp4_quantizer(enable_quant=True, enable_calib=False)
        amax = weight.detach().float().abs().amax()
        if amax.ndim == 0:
            amax = amax.unsqueeze(0)
        wq.register_buffer("_amax", amax.to(dev))
        module.weight_quantizer = wq

        iq = _make_nvfp4_quantizer(enable_quant=True, enable_calib=False)
        iq.register_buffer("_amax", torch.tensor([0.5], device=dev))
        module.input_quantizer = iq

        count += 1
        if count <= 3:
            log.info(
                "  Injected NVFP4 quantizers on %s  weight=%s  dev=%s  "
                "amax=%.4f  wq._calibrator=%s  iq._calibrator=%s",
                name, list(weight.shape), dev, amax.item(),
                type(wq._calibrator).__name__ if getattr(wq, "_calibrator", None) else "MISSING",
                type(iq._calibrator).__name__ if getattr(iq, "_calibrator", None) else "MISSING",
            )
        elif count == 4:
            log.info("  ... (suppressing further MoELinear injection logs)")

    return count


def _ensure_weight_quantizer_calibrated(model: torch.nn.Module) -> int:
    """Fill in ``_amax`` on any weight quantizer left un-calibrated (e.g. MoE
    gate weights or experts the router never activated).  Derives ``_amax``
    from the weight tensor itself.  Returns the number patched.
    """
    patched = 0
    for name, module in model.named_modules():
        wq = getattr(module, "weight_quantizer", None)
        if wq is None:
            continue
        if hasattr(wq, "_amax") and wq._amax is not None:
            continue

        weight = getattr(module, "weight", None)
        if weight is None:
            continue

        amax = weight.detach().float().abs().amax()
        if amax.ndim == 0:
            amax = amax.unsqueeze(0)

        wq.register_buffer("_amax", amax)
        patched += 1
        log.info("  Patched %s.weight_quantizer  _amax=%.4f", name, amax.max().item())

    return patched


class _ExpertCoverageTracker:
    """Hooks into Step3p5MoEMLP to count expert activation during calibration."""

    def __init__(self, model: torch.nn.Module):
        self.expert_hits: dict[str, torch.Tensor] = {}
        self._hooks: list = []

        for name, module in model.named_modules():
            if "Step3p5MoEMLP" not in type(module).__name__:
                continue
            num_experts = getattr(module, "num_experts", None)
            if num_experts is None:
                continue
            counter = torch.zeros(num_experts, dtype=torch.long)
            self.expert_hits[name] = counter

            def _make_hook(counter_ref):
                def _hook(mod, inputs, output):
                    hidden = inputs[0]
                    gate = getattr(mod, "gate", None)
                    if gate is None:
                        return
                    with torch.no_grad():
                        router_logits = gate(hidden.view(-1, hidden.shape[-1]))
                        top_k = getattr(mod, "top_k", 8)
                        _, selected = torch.topk(router_logits, top_k, dim=-1)
                        for idx in selected.view(-1).cpu():
                            counter_ref[idx.item()] += 1
                return _hook

            h = module.register_forward_hook(_make_hook(counter))
            self._hooks.append(h)

    def report_and_cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for name, counter in self.expert_hits.items():
            hit = (counter > 0).sum().item()
            num_experts = len(counter)
            min_hits = counter.min().item()
            max_hits = counter.max().item()
            zero_experts = num_experts - hit
            log.info(
                "  Expert coverage '%s': %d/%d hit (%.0f%%), min=%d, max=%d, %d never activated",
                name, hit, num_experts, 100.0 * hit / max(num_experts, 1),
                min_hits, max_hits, zero_experts,
            )
            if zero_experts > 0:
                log.warning(
                    "  %d/%d experts in '%s' never activated during calibration. "
                    "Consider increasing --calib-size.",
                    zero_experts, num_experts, name,
                )


# =========================================================================
# Stage 1 — Download & Load Model to GPU (+ unified memory overflow)
# =========================================================================
def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load Step-3.5-Flash in bf16 across 3× B200 GPUs.

    Uses ``device_map="auto"`` so Accelerate distributes layers across all 3
    GPUs.  The ~392 GB bf16 model fits within the combined ~576 GB VRAM,
    so CPU offload should be minimal or unnecessary.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    num_gpus = torch.cuda.device_count()
    total_vram_gb = sum(
        torch.cuda.get_device_properties(i).total_memory / (1024**3)
        for i in range(num_gpus)
    ) if num_gpus > 0 else 0.0

    log.info(
        "Detected %d GPU(s), %.0f GB total VRAM.  Loading %s in bf16 …",
        num_gpus, total_vram_gb, model_id,
    )
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs detected")

    _log_system_memory("before model load")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    device_map = getattr(model, "hf_device_map", {})
    if device_map:
        devices_used = set(str(v) for v in device_map.values())
        cpu_layers = sum(1 for v in device_map.values() if str(v) == "cpu")
        gpu_layers = len(device_map) - cpu_layers
        log.info(
            "Model loaded across devices: %s  (%d layers on GPU, %d on CPU/unified mem)",
            sorted(devices_used), gpu_layers, cpu_layers,
        )
    else:
        log.info("Model loaded.  device_map: n/a")

    _log_gpu_memory("after model load")
    _log_system_memory("after model load")
    return model, tokenizer


# =========================================================================
# Stage 2 — ModelOpt NVFP4 Post-Training Quantization
# =========================================================================
def _collect_texts_from_source(
    dataset_id: str,
    config: str | None,
    field: str,
    target: int,
    min_chars: int = 100,
) -> list[str]:
    """Stream up to *target* texts from a single HuggingFace dataset source."""
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_id, config, split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset(dataset_id, config, split="test", streaming=True)
        except Exception as exc:
            log.warning("  ✗ %s — failed to load: %s", dataset_id, exc)
            return []

    texts: list[str] = []
    for sample in ds:
        text = sample.get(field) or sample.get("text", "")
        if len(text.strip()) >= min_chars:
            texts.append(text)
        if len(texts) >= target:
            break

    log.info("  ✓ %-35s  %d / %d samples", dataset_id, len(texts), target)
    return texts


def build_calibration_dataloader(
    tokenizer,
    sources: list[tuple[str, str | None, str, int]],
    total_samples: int,
    max_seq_len: int,
    batch_size: int,
):
    """Build a multi-source calibration dataloader from diverse HF datasets
    so the MoE router activates as many experts as possible.
    """
    import random

    log.info(
        "Collecting calibration data from %d source(s)  "
        "(target=%d samples, max_seq_len=%d) …",
        len(sources), total_samples, max_seq_len,
    )

    all_texts: list[str] = []
    remaining = total_samples
    succeeded_sources: list[tuple[str, str | None, str]] = []

    for dataset_id, config, field, cap in sources:
        want = min(cap, remaining)
        if want <= 0:
            continue
        texts = _collect_texts_from_source(dataset_id, config, field, want)
        all_texts.extend(texts)
        remaining -= len(texts)
        if texts:
            succeeded_sources.append((dataset_id, config, field))

    if remaining > 0 and succeeded_sources:
        log.info(
            "Short by %d sample(s) — back-filling from successful source(s) …",
            remaining,
        )
        for dataset_id, config, field in succeeded_sources:
            if remaining <= 0:
                break
            extra = _collect_texts_from_source(dataset_id, config, field, remaining)
            all_texts.extend(extra)
            remaining -= len(extra)

    random.seed(42)
    random.shuffle(all_texts)
    all_texts = all_texts[:total_samples]
    log.info("Calibration corpus: %d samples from %d source(s).", len(all_texts), len(sources))

    tokenized = [
        tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )["input_ids"].squeeze(0)
        for text in all_texts
    ]

    import torch.utils.data as data_utils

    class _CalibDataset(data_utils.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return {"input_ids": self.samples[idx]}

    dataset = _CalibDataset(tokenized)
    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def _patch_max_calibrator_for_bf16():
    """Patch MaxCalibrator.collect to handle bf16 NaN/Inf without asserting."""
    try:
        from modelopt.torch.quantization.calib.max import MaxCalibrator
        from modelopt.torch.quantization import utils as quant_utils
    except ImportError:
        return False

    @torch.no_grad()
    def _safe_collect(self, x):
        if x.dtype == torch.bfloat16:
            x = x.float()

        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()

        if x.device.type == "meta":
            self._calib_amax = local_amax
            return

        local_amax = local_amax.clamp(min=0)
        local_amax = torch.where(torch.isfinite(local_amax), local_amax, torch.zeros_like(local_amax))

        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax = torch.max(self._calib_amax, local_amax)

        if self._track_amax:
            self._amaxs.append(local_amax.cpu().numpy())

    MaxCalibrator.collect = _safe_collect
    log.info("Patched MaxCalibrator.collect — NaN/Inf/bf16 safe.")
    return True


def quantize_model(
    model,
    tokenizer,
    calib_size: int = CALIB_SIZE,
    calib_max_seq_len: int = CALIB_MAX_SEQ_LEN,
    calib_batch_size: int = CALIB_BATCH_SIZE,
):
    """Run ModelOpt NVFP4 PTQ on the loaded model (in-place)."""
    import modelopt.torch.quantization as mtq

    _patch_max_calibrator_for_bf16()

    log.info("Building multi-source calibration dataloader …")
    calib_loader = build_calibration_dataloader(
        tokenizer,
        sources=CALIB_SOURCES,
        total_samples=calib_size,
        max_seq_len=calib_max_seq_len,
        batch_size=calib_batch_size,
    )

    coverage = _ExpertCoverageTracker(model)

    def forward_loop(model):
        import time
        total = len(calib_loader)
        embed_device = next(model.parameters()).device
        log.info("Running calibration forward passes (%d batches, input→%s) …", total, embed_device)
        t_start = time.time()
        t_batch_start = t_start

        for i, batch in enumerate(calib_loader):
            seq_len = batch["input_ids"].shape[-1]
            input_ids = batch["input_ids"].to(embed_device)
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)

            t_now = time.time()
            batch_sec = t_now - t_batch_start
            elapsed = t_now - t_start
            done = i + 1

            if done == 1 or done % 10 == 0 or done == total:
                avg = elapsed / done
                eta = avg * (total - done)
                eta_m, eta_s = divmod(int(eta), 60)
                log.info(
                    "  batch %4d / %d  | seq_len=%4d | %.1fs/batch | elapsed %.0fs | ETA %dm%02ds",
                    done, total, seq_len, batch_sec, elapsed, eta_m, eta_s,
                )
            t_batch_start = t_now

    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    log.info("Starting NVFP4 quantization via ModelOpt …")
    _log_gpu_memory("before quantization")

    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    log.info("Quantization complete.")
    log.info("[quantize] Patching calibrators post-quantization …")
    _patch_tensorquantizer_calibrator_compat(model)
    log.info("[quantize] Reporting expert coverage …")
    coverage.report_and_cleanup()

    mtq.print_quant_summary(model)
    _log_gpu_memory("after quantization")
    _log_system_memory("after quantization")
    return model


def save_quantized_state(model, save_path: str):
    """Save the quantized model state dict so calibration can be resumed."""
    import modelopt.torch.opt as mto

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = save_path / "quantized_state.pt"

    log.info("Saving quantized model state to %s …", ckpt_file)
    mto.save(model, str(ckpt_file))
    log.info("Quantized state saved (%.1f GB).", ckpt_file.stat().st_size / (1024**3))


def load_quantized_state(model, save_path: str):
    """Restore a previously saved quantized model state (skip calibration)."""
    import modelopt.torch.opt as mto

    ckpt_file = Path(save_path) / "quantized_state.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"No quantized state found at {ckpt_file}")

    _patch_tensorquantizer_calibrator_compat(model)
    log.info("Loading quantized state from %s …", ckpt_file)
    model = mto.restore(model, str(ckpt_file))
    _patch_tensorquantizer_calibrator_compat(model)
    log.info("Quantized state restored — skipping calibration.")
    return model


# =========================================================================
# Stage 3 — Export
# =========================================================================
def export_checkpoint(model, tokenizer, export_dir: str):
    """Export the quantized model as an NVFP4 HuggingFace checkpoint.

    Patches missing ``_amax`` values on uncalibrated weight quantizers,
    injects NVFP4 quantizers onto MoELinear modules, then delegates to
    ModelOpt's ``export_hf_checkpoint``.
    """
    from modelopt.torch.export import export_hf_checkpoint

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    log.info("[export] Step 1/5: Patching TensorQuantizer calibrator compatibility …")
    _patch_tensorquantizer_calibrator_compat(model)

    log.info("[export] Step 2/5: Ensuring all weight quantizers have _amax …")
    patched = _ensure_weight_quantizer_calibrated(model)
    if patched:
        log.info("Fixed %d weight quantizer(s) missing _amax.", patched)

    log.info("[export] Step 3/5: Injecting NVFP4 quantizers onto MoELinear modules …")
    injected = _inject_moe_linear_quantizers(model)
    if injected:
        log.info("Injected NVFP4 quantizers on %d MoELinear modules.", injected)

    log.info("[export] Step 4/5: Re-patching calibrators after injection …")
    _patch_tensorquantizer_calibrator_compat(model)

    log.info("[export] Step 5/5: Calling export_hf_checkpoint → %s …", export_path)
    export_hf_checkpoint(model, export_dir=str(export_path))
    tokenizer.save_pretrained(str(export_path))

    log.info("Export finished.  Contents of %s:", export_path)
    for f in sorted(export_path.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        log.info("  %s  (%.1f MB)", f.name, size_mb)


def cleanup_hf_cache(model_id: str = MODEL_ID):
    """Delete the HuggingFace disk cache for the BF16 model to free ~392 GB."""
    from huggingface_hub import scan_cache_dir

    log.info("Scanning HF cache for %s …", model_id)
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == model_id:
            for revision in repo.revisions:
                log.info(
                    "Deleting cached revision %s  (%.1f GB) …",
                    revision.commit_hash[:12],
                    revision.size_on_disk / (1024**3),
                )
                strategy = cache_info.delete_revisions(revision.commit_hash)
                strategy.execute()
            log.info("BF16 cache cleaned up.")
            return
    log.warning("Model %s not found in HF cache — nothing to clean.", model_id)


# =========================================================================
# Stage 4 — Serve Inference via vLLM
# =========================================================================
def serve_vllm(export_dir: str, tp_size: int = 0):
    """Launch a vLLM OpenAI-compatible server with the NVFP4 checkpoint."""
    tp = _resolve_tp_size(tp_size)
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(export_dir),
        "--quantization", "modelopt",
        "--tensor-parallel-size", str(tp),
        "--trust-remote-code",
        "--reasoning-parser", "step3p5",
        "--tool-call-parser", "step3p5",
        "--enable-auto-tool-choice",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    log.info("Launching vLLM server (TP=%d):\n  %s", tp, " ".join(cmd))
    print(
        textwrap.dedent("""\
        ---------------------------------------------------------------
        vLLM server starting on http://0.0.0.0:8000

        Quick test with curl:
            curl http://localhost:8000/v1/chat/completions \\
              -H "Content-Type: application/json" \\
              -d '{
                "model": "%s",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 256
              }'

        Or with Python:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
            resp = client.chat.completions.create(
                model="%s",
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=256,
            )
            print(resp.choices[0].message.content)
        ---------------------------------------------------------------
        """)
        % (export_dir, export_dir)
    )
    subprocess.run(cmd, check=True)


# =========================================================================
# Helpers — Logging
# =========================================================================
def _log_gpu_memory(label: str = ""):
    """Print per-GPU memory usage."""
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        log.info(
            "  GPU %d  %s: %.1f GB allocated / %.1f GB reserved / %.1f GB total",
            i, label, alloc, reserved, total,
        )


def _log_system_memory(label: str = ""):
    """Print system (CPU / unified) memory usage."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        log.info(
            "  System RAM %s: %.1f GB used / %.1f GB available / %.1f GB total",
            label,
            vm.used / (1024**3),
            vm.available / (1024**3),
            vm.total / (1024**3),
        )
    except ImportError:
        pass


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="StepFun-3.5-Flash → NVFP4 quantization & vLLM serving (3× B200)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Full pipeline (download + quantize + export)
          python run_stepfun_nvfp4.py --export-dir ./stepfun-nvfp4-ckpt

          # Full pipeline + reclaim BF16 cache disk space
          python run_stepfun_nvfp4.py --export-dir ./stepfun-nvfp4-ckpt --cleanup-cache

          # Launch vLLM from an existing NVFP4 checkpoint
          python run_stepfun_nvfp4.py --serve-only --export-dir ./stepfun-nvfp4-ckpt

          # Point HF cache to a bigger volume
          HF_HOME=/mnt/nvme/hf_cache python run_stepfun_nvfp4.py --export-dir ./out
        """),
    )
    p.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="HuggingFace model ID (default: %(default)s)",
    )
    p.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help="Directory for the exported NVFP4 checkpoint (default: %(default)s)",
    )
    p.add_argument(
        "--calib-size",
        type=int,
        default=CALIB_SIZE,
        help="Number of calibration samples for PTQ (default: %(default)s)",
    )
    p.add_argument(
        "--calib-max-seq-len",
        type=int,
        default=CALIB_MAX_SEQ_LEN,
        help="Max sequence length for calibration tokens (default: %(default)s)",
    )
    p.add_argument(
        "--tp-size",
        type=int,
        default=TP_SIZE,
        help="Tensor-parallel size for vLLM serving; 0 = auto-detect from GPU count (default: %(default)s)",
    )
    p.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Delete the BF16 HF cache after successful export to reclaim ~392 GB disk",
    )
    p.add_argument(
        "--serve-only",
        action="store_true",
        help="Skip quantization; launch vLLM from an existing --export-dir",
    )
    p.add_argument(
        "--no-serve",
        action="store_true",
        help="Run quantization + export only — do NOT start the vLLM server",
    )
    p.add_argument(
        "--quantized-state-dir",
        default=None,
        help="Directory for saving/loading intermediate quantized state. "
             "After calibration, state is auto-saved here. Use with "
             "--resume-quantized to skip calibration on re-runs.",
    )
    p.add_argument(
        "--resume-quantized",
        action="store_true",
        help="Skip calibration — load previously saved quantized state from "
             "--quantized-state-dir and go straight to export.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    effective_tp = _resolve_tp_size(args.tp_size)

    log.info("=" * 65)
    log.info("StepFun-3.5-Flash  →  NVFP4 Pipeline  (3× B200)")
    log.info("=" * 65)
    log.info("  Model ID       : %s", args.model_id)
    log.info("  Export dir      : %s", args.export_dir)
    log.info("  TP size         : %d  (requested=%d)", effective_tp, args.tp_size)
    log.info("  Calib samples   : %d  (from %d source(s))", args.calib_size, len(CALIB_SOURCES))
    log.info("  Calib sources   : %s", ", ".join(s[0] for s in CALIB_SOURCES))
    log.info("  Serve only      : %s", args.serve_only)
    log.info("  No serve        : %s", args.no_serve)
    log.info("  Cleanup cache   : %s", args.cleanup_cache)
    log.info("  HF_HOME         : %s", os.environ.get("HF_HOME", "(default)"))
    log.info("  GPUs available  : %d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        log.info("    GPU %d: %s  (%.0f GB)", i, props.name, props.total_memory / (1024**3))
    _log_system_memory("at startup")
    log.info("=" * 65)

    # ------------------------------------------------------------------
    # Serve-only shortcut
    # ------------------------------------------------------------------
    if args.serve_only:
        if not Path(args.export_dir).exists():
            log.error("--serve-only specified but export dir does not exist: %s", args.export_dir)
            sys.exit(1)
        serve_vllm(args.export_dir, tp_size=args.tp_size)
        return

    # ------------------------------------------------------------------
    # Stage 1 — Load
    # ------------------------------------------------------------------
    log.info("[Stage 1/3]  Downloading & loading model across 3× B200 GPUs …")
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    _patch_tensorquantizer_calibrator_compat(model)

    # ------------------------------------------------------------------
    # Stage 2 — Quantize (or resume from saved state)
    # ------------------------------------------------------------------
    quant_state_dir = args.quantized_state_dir or os.path.join(args.export_dir, "_quant_state")

    if args.resume_quantized:
        log.info("[Stage 2/3]  Resuming from saved quantized state …")
        model = load_quantized_state(model, quant_state_dir)
    else:
        log.info("[Stage 2/3]  NVFP4 Post-Training Quantization …")
        model = quantize_model(
            model, tokenizer,
            calib_size=args.calib_size,
            calib_max_seq_len=args.calib_max_seq_len,
        )
        try:
            save_quantized_state(model, quant_state_dir)
        except Exception as e:
            log.warning("Could not save quantized state (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Stage 3 — Export
    # ------------------------------------------------------------------
    log.info("[Stage 3/3]  Exporting NVFP4 checkpoint …")
    export_checkpoint(model, tokenizer, args.export_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Model unloaded from GPU.  Memory released.")

    if args.cleanup_cache:
        log.info("Cleaning HF cache to free ~392 GB disk …")
        cleanup_hf_cache(args.model_id)

    log.info("Pipeline complete.  NVFP4 checkpoint at: %s", args.export_dir)

    # ------------------------------------------------------------------
    # Stage 4 — Serve (unless --no-serve)
    # ------------------------------------------------------------------
    if not args.no_serve:
        serve_vllm(args.export_dir, tp_size=args.tp_size)


if __name__ == "__main__":
    main()