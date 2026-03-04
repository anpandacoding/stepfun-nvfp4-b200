#!/usr/bin/env python3
"""Post-process NVFP4 checkpoint: fix dense INT8 layers and align MoE w1/w3 scales.

Two critical fixes for vLLM MIXED_PRECISION inference:

1. Dense layers (0-2): W8A8_SQ_PER_CHANNEL INT8 weights are dequantized back to
   bf16. vLLM's MIXED_PRECISION handler doesn't support W8A8, so INT8 bytes get
   interpreted as bf16 = garbage from layer 0 onwards.

2. MoE layers (3-44): gate_proj (w1) and up_proj (w3) weight_scale_2 values are
   aligned to a shared max per expert. vLLM fuses w1+w3 into w13 requiring
   matching scale_2; independent scales cause dequantization errors across all
   42 layers × 288 expert computations.

Works on both fresh (bf16 MoE) and already-repacked (uint8 MoE) checkpoints.

Usage:
    python repack_moe_fp4.py --ckpt-dir ./stepfun-nvfp4-ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BLOCK_SIZE = 16
DEFAULT_INPUT_SCALE = torch.tensor(0.5 / (6.0 * 448.0), dtype=torch.float32)


# =========================================================================
# Detection helpers
# =========================================================================

def is_moe_bf16_weight(name: str, tensor: torch.Tensor) -> bool:
    """Identify bf16 3D MoE expert weight tensors that need NVFP4 packing."""
    return (
        tensor.dtype == torch.bfloat16
        and tensor.ndim == 3
        and "moe" in name.lower()
        and "weight" in name.lower()
        and "scale" not in name.lower()
        and "norm" not in name.lower()
        and ".gate." not in name.lower()
        and "share_expert" not in name.lower()
    )


def is_dense_int8_weight(name: str, tensor: torch.Tensor) -> bool:
    """Identify INT8 dense MLP weights from W8A8_SQ_PER_CHANNEL export."""
    return (
        tensor.dtype == torch.int8
        and tensor.ndim == 2
        and "mlp" in name.lower()
        and name.endswith(".weight")
        and "scale" not in name.lower()
    )


def _parse_layer_and_proj(name: str) -> tuple[int, str] | None:
    """Extract (layer_idx, proj_type) from a MoE tensor name."""
    m = re.search(r"layers\.(\d+)\.moe\.(gate_proj|up_proj|down_proj)", name)
    if m:
        return int(m.group(1)), m.group(2)
    return None


# =========================================================================
# Quantization
# =========================================================================

def quantize_tensor_nvfp4(weight: torch.Tensor, forced_scale_2: torch.Tensor | None = None):
    """Quantize a 3D MoE weight tensor to NVFP4, per-expert.

    If forced_scale_2 [num_experts] is provided, each expert uses the given
    global scale instead of computing its own.
    """
    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_experts = weight.shape[0]

    packed_list, scale_list, scale_2_list = [], [], []

    for i in range(num_experts):
        w = weight[i].to(device).float()
        fs2 = forced_scale_2[i:i+1].to(device) if forced_scale_2 is not None else None
        qtensor, w_scale, w_scale_2 = NVFP4QTensor.quantize(
            w,
            block_size=BLOCK_SIZE,
            weights_scaling_factor=None,
            weights_scaling_factor_2=fs2,
        )
        packed_list.append(qtensor._quantized_data.cpu())
        scale_list.append(w_scale.cpu())
        scale_2_list.append(w_scale_2.cpu().reshape(1))

    packed = torch.stack(packed_list)
    w_scale = torch.stack(scale_list)
    w_scale_2 = torch.cat(scale_2_list)

    return packed, w_scale, w_scale_2


def dequantize_int8_perchannel(weight_int8: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize per-channel INT8 weight back to bf16."""
    scale = weight_scale.float()
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    return (weight_int8.float() * scale).to(torch.bfloat16)


def adjust_moe_scale_2(
    weight_scale: torch.Tensor,
    old_scale_2: torch.Tensor,
    new_scale_2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust weight_scale when changing weight_scale_2 for already-packed NVFP4.

    dequant = packed_fp4 * weight_scale * scale_2
    To keep dequant invariant: new_weight_scale = weight_scale * (old_s2 / new_s2)

    Returns (adjusted_weight_scale, new_scale_2).
    """
    ratio = old_scale_2.float() / new_scale_2.float().clamp(min=1e-12)
    adjusted = weight_scale.float() * ratio[:, None, None]
    return adjusted.to(weight_scale.dtype), new_scale_2


# =========================================================================
# Calibration data loading
# =========================================================================

def _load_calibrated_input_amax(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load per-expert input amax from calibration, if available."""
    calib_path = ckpt_dir / "_calib" / "expert_input_amax.pt"
    if not calib_path.exists():
        log.warning("No calibrated expert_input_amax.pt found — using fallback.")
        return {}
    data = torch.load(str(calib_path), map_location="cpu", weights_only=True)
    log.info("Loaded calibrated input_amax for %d MoELinear projections.", len(data))
    return data


def _get_input_scale(calib_amax: dict[str, torch.Tensor], tensor_name: str, num_experts: int) -> torch.Tensor:
    """Get per-expert input_scale [num_experts] from calibration data."""
    module_name = tensor_name.replace(".weight", "")
    if module_name in calib_amax:
        amax = calib_amax[module_name]
        scale = amax.clamp(min=1e-12).float()
        calibrated = (scale > 0).sum().item()
        if calibrated > 0:
            median_scale = scale[scale > 0].median()
            scale[scale == 0] = median_scale
            return scale
    return DEFAULT_INPUT_SCALE.expand(num_experts).clone()


# =========================================================================
# Phase 1: Compute shared w1/w3 scale_2
# =========================================================================

def collect_moe_scale_2(ckpt_dir: Path, shards: list[Path]) -> dict[int, dict[str, torch.Tensor]]:
    """Read weight_scale_2 for all MoE gate_proj/up_proj across layers.

    For already-packed shards, reads the existing weight_scale_2.
    For fresh (bf16) shards, computes natural scale_2 via quantization.

    Returns {layer_idx: {"gate_proj": Tensor[288], "up_proj": Tensor[288]}}.
    """
    layer_scales: dict[int, dict[str, torch.Tensor]] = {}

    for shard_path in shards:
        with safe_open(str(shard_path), framework="pt") as f:
            keys = list(f.keys())
            for key in keys:
                parsed = _parse_layer_and_proj(key)
                if parsed is None:
                    continue
                layer_idx, proj_type = parsed
                if proj_type not in ("gate_proj", "up_proj"):
                    continue

                s2_key = key.replace(".weight", ".weight_scale_2")
                if s2_key in keys and key.endswith(".weight"):
                    s2 = f.get_tensor(s2_key)
                    log.info("  Read scale_2 for layer %d %s: shape=%s", layer_idx, proj_type, list(s2.shape))
                elif key.endswith(".weight"):
                    t = f.get_tensor(key)
                    if is_moe_bf16_weight(key, t):
                        log.info("  Computing scale_2 for layer %d %s (fresh bf16) …", layer_idx, proj_type)
                        _, _, s2 = quantize_tensor_nvfp4(t)
                    else:
                        continue
                else:
                    continue

                if layer_idx not in layer_scales:
                    layer_scales[layer_idx] = {}
                layer_scales[layer_idx][proj_type] = s2

    return layer_scales


def compute_shared_scale_2(layer_scales: dict[int, dict[str, torch.Tensor]]) -> dict[int, torch.Tensor]:
    """Compute shared_scale_2 = max(gate_s2, up_s2) per expert per layer."""
    shared: dict[int, torch.Tensor] = {}
    for layer_idx in sorted(layer_scales.keys()):
        scales = layer_scales[layer_idx]
        if "gate_proj" in scales and "up_proj" in scales:
            shared_s2 = torch.max(scales["gate_proj"], scales["up_proj"])
            shared[layer_idx] = shared_s2
            diff = (scales["gate_proj"] - scales["up_proj"]).abs().mean().item()
            log.info("  Layer %d: shared scale_2 computed (mean |gate-up| diff: %.6f)", layer_idx, diff)
        elif "gate_proj" in scales:
            shared[layer_idx] = scales["gate_proj"]
        elif "up_proj" in scales:
            shared[layer_idx] = scales["up_proj"]
    return shared


# =========================================================================
# Phase 2: Repack shards
# =========================================================================

def repack_shard(
    shard_path: Path,
    shared_scale_2: dict[int, torch.Tensor],
    layer_scales: dict[int, dict[str, torch.Tensor]],
    calib_amax: dict[str, torch.Tensor] | None = None,
) -> dict[str, str]:
    """Repack a single shard, applying all fixes. Returns weight_map entries."""
    if calib_amax is None:
        calib_amax = {}
    log.info("Processing %s …", shard_path.name)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(shard_path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    new_tensors: dict[str, torch.Tensor] = {}
    skip_keys: set[str] = set()
    changes = 0

    # --- Fix 1: Dequantize dense INT8 weights ---
    for name in sorted(tensors.keys()):
        t = tensors[name]
        if is_dense_int8_weight(name, t):
            scale_key = name.replace(".weight", ".weight_scale")
            input_scale_key = name.replace(".weight", ".input_scale")
            if scale_key in tensors:
                bf16_weight = dequantize_int8_perchannel(t, tensors[scale_key])
                new_tensors[name] = bf16_weight
                skip_keys.add(scale_key)
                if input_scale_key in tensors:
                    skip_keys.add(input_scale_key)
                changes += 1
                log.info(
                    "  %-60s INT8 → bf16  shape=%s",
                    name, list(bf16_weight.shape),
                )
            else:
                log.warning("  %s is INT8 but no weight_scale found — keeping as-is", name)
                new_tensors[name] = t

    # --- Fix 2a: Repack fresh bf16 MoE weights with shared scale_2 ---
    # --- Fix 2b: Adjust already-packed MoE scale_2 to shared ---
    for name in sorted(tensors.keys()):
        if name in new_tensors or name in skip_keys:
            continue

        t = tensors[name]

        if is_moe_bf16_weight(name, t):
            bf16_mb = t.nelement() * t.element_size() / 1e6
            t0 = time.time()

            parsed = _parse_layer_and_proj(name)
            forced_s2 = None
            if parsed is not None:
                layer_idx, proj_type = parsed
                if proj_type in ("gate_proj", "up_proj") and layer_idx in shared_scale_2:
                    forced_s2 = shared_scale_2[layer_idx]

            packed, w_scale, w_scale_2 = quantize_tensor_nvfp4(t, forced_scale_2=forced_s2)
            elapsed = time.time() - t0
            packed_mb = packed.nelement() * packed.element_size() / 1e6

            new_tensors[name] = packed
            new_tensors[name.replace(".weight", ".weight_scale")] = w_scale
            new_tensors[name.replace(".weight", ".weight_scale_2")] = w_scale_2
            num_experts = t.shape[0]
            input_scale = _get_input_scale(calib_amax, name, num_experts)
            new_tensors[name.replace(".weight", ".input_scale")] = input_scale

            skip_keys.update([
                name.replace(".weight", ".weight_scale"),
                name.replace(".weight", ".weight_scale_2"),
                name.replace(".weight", ".input_scale"),
            ])

            changes += 1
            shared_str = " [shared w13 scale]" if forced_s2 is not None else ""
            log.info(
                "  %-60s bf16 %.0f MB → uint8 %.0f MB%s  (%.1fs)",
                name, bf16_mb, packed_mb, shared_str, elapsed,
            )
            continue

        # Already-packed uint8 MoE weight — check if scale_2 needs alignment
        if (
            t.dtype == torch.uint8
            and t.ndim == 3
            and name.endswith(".weight")
            and "moe" in name
            and "scale" not in name
        ):
            parsed = _parse_layer_and_proj(name)
            if parsed is not None:
                layer_idx, proj_type = parsed
                if proj_type in ("gate_proj", "up_proj") and layer_idx in shared_scale_2:
                    s2_key = name.replace(".weight", ".weight_scale_2")
                    scale_key = name.replace(".weight", ".weight_scale")
                    old_s2 = tensors.get(s2_key)
                    old_scale = tensors.get(scale_key)
                    new_s2 = shared_scale_2[layer_idx]

                    if old_s2 is not None and old_scale is not None:
                        if not torch.equal(old_s2, new_s2):
                            adjusted_scale, final_s2 = adjust_moe_scale_2(old_scale, old_s2, new_s2)
                            new_tensors[name] = t
                            new_tensors[scale_key] = adjusted_scale
                            new_tensors[s2_key] = final_s2
                            skip_keys.update([scale_key, s2_key])
                            max_ratio_diff = (old_s2.float() / new_s2.float().clamp(min=1e-12) - 1.0).abs().max().item()
                            changes += 1
                            log.info(
                                "  %-60s scale_2 aligned (max ratio diff: %.4f)",
                                name, max_ratio_diff,
                            )
                            continue

        if name not in new_tensors and name not in skip_keys:
            new_tensors[name] = t

    if changes == 0:
        log.info("  No changes needed.")
        return {k: shard_path.name for k in new_tensors}

    save_file(new_tensors, str(shard_path))
    new_size_mb = shard_path.stat().st_size / 1e6
    log.info("  Rewrote %s: %d fix(es), new size %.0f MB", shard_path.name, changes, new_size_mb)
    return {k: shard_path.name for k in new_tensors}


# =========================================================================
# Index & config updates
# =========================================================================

def update_index(ckpt_dir: Path, full_weight_map: dict[str, str]):
    """Rewrite the safetensors index with updated weight map."""
    idx_path = ckpt_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        log.warning("No index file found — skipping index update.")
        return

    with open(idx_path) as f:
        idx = json.load(f)

    idx["weight_map"] = dict(sorted(full_weight_map.items()))

    total_size = 0
    for shard_name in set(full_weight_map.values()):
        shard_path = ckpt_dir / shard_name
        if shard_path.exists():
            total_size += shard_path.stat().st_size
    idx["metadata"] = {"total_size": total_size}

    with open(idx_path, "w") as f:
        json.dump(idx, f, indent=2)
    log.info("Updated index: %d keys, total size %.1f GB", len(full_weight_map), total_size / 1e9)


def fix_quant_config(ckpt_dir: Path):
    """Remove W8A8_SQ_PER_CHANNEL entries from hf_quant_config.json.

    Dense layers are now bf16 so these entries would confuse vLLM.
    """
    config_path = ckpt_dir / "hf_quant_config.json"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = json.load(f)

    quant_layers = config.get("quantization", {}).get("quantized_layers", {})
    orig_count = len(quant_layers)
    cleaned = {
        k: v for k, v in quant_layers.items()
        if v.get("quant_algo") != "W8A8_SQ_PER_CHANNEL"
    }
    removed = orig_count - len(cleaned)

    if removed:
        config["quantization"]["quantized_layers"] = cleaned
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info("Removed %d W8A8_SQ_PER_CHANNEL entries from hf_quant_config.json", removed)
    else:
        log.info("No W8A8 entries found in hf_quant_config.json — no changes.")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fix NVFP4 checkpoint: dequant dense INT8 + align MoE w1/w3 scales",
    )
    parser.add_argument("--ckpt-dir", required=True, help="Path to the NVFP4 checkpoint")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    shards = sorted(ckpt_dir.glob("model-*.safetensors"))
    log.info("Found %d shard(s) in %s", len(shards), ckpt_dir)

    # Phase 1: Collect and compute shared scale_2
    log.info("=" * 65)
    log.info("Phase 1: Collecting MoE gate_proj/up_proj scale_2 values …")
    log.info("=" * 65)
    layer_scales = collect_moe_scale_2(ckpt_dir, shards)
    log.info("Found scale_2 data for %d MoE layers.", len(layer_scales))

    shared_s2 = compute_shared_scale_2(layer_scales)
    log.info("Computed shared scale_2 for %d layers.", len(shared_s2))

    # Phase 2: Repack all shards
    log.info("=" * 65)
    log.info("Phase 2: Repacking shards …")
    log.info("  Fix 1: Dense INT8 MLP → bf16 (layers 0-2)")
    log.info("  Fix 2: MoE gate_proj/up_proj → shared weight_scale_2")
    log.info("=" * 65)

    calib_amax = _load_calibrated_input_amax(ckpt_dir)

    full_weight_map: dict[str, str] = {}
    total_t0 = time.time()

    for shard_path in shards:
        shard_map = repack_shard(shard_path, shared_s2, layer_scales, calib_amax)
        full_weight_map.update(shard_map)

    # Phase 3: Update index and config
    log.info("=" * 65)
    log.info("Phase 3: Updating index and quant config …")
    log.info("=" * 65)
    update_index(ckpt_dir, full_weight_map)
    fix_quant_config(ckpt_dir)

    elapsed = time.time() - total_t0
    log.info("Done in %.0f seconds.", elapsed)

    total_gb = sum(s.stat().st_size for s in shards if s.exists()) / 1e9
    log.info("Total checkpoint size: %.1f GB", total_gb)


if __name__ == "__main__":
    main()
