#!/usr/bin/env python3
"""Post-process NVFP4 checkpoint: pack bf16 MoE expert weights to FP4.

Reads each safetensors shard, finds bf16 3D MoE weight tensors that the
ModelOpt exporter missed, quantizes them using ModelOpt's own NVFP4QTensor,
and rewrites the shards in-place.  Updates the safetensors index afterwards.

Usage:
    python repack_moe_fp4.py --ckpt-dir ./stepfun-nvfp4-ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
# Default activation scale for MoE expert inputs (conservative fallback)
DEFAULT_INPUT_SCALE = torch.tensor(0.5 / (6.0 * 448.0), dtype=torch.float32)


def is_moe_bf16_weight(name: str, tensor: torch.Tensor) -> bool:
    """Identify bf16 3D MoE expert weight tensors that need repacking."""
    return (
        tensor.dtype == torch.bfloat16
        and tensor.ndim == 3
        and "moe" in name.lower()
        and "weight" in name.lower()
        and "scale" not in name.lower()
        and "norm" not in name.lower()
    )


def quantize_tensor_nvfp4(weight: torch.Tensor):
    """Quantize a tensor to NVFP4 using ModelOpt's own packing code.

    Returns (packed_uint8, weight_scale_fp8, weight_scale_2_fp32).
    """
    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = weight.to(device).float()

    qtensor, w_scale, w_scale_2 = NVFP4QTensor.quantize(
        w,
        block_size=BLOCK_SIZE,
        weights_scaling_factor=None,
        weights_scaling_factor_2=None,
    )

    packed = qtensor._quantized_data.cpu()
    w_scale = w_scale.cpu()
    w_scale_2 = w_scale_2.cpu()

    return packed, w_scale, w_scale_2


def repack_shard(shard_path: Path) -> dict[str, str]:
    """Repack a single safetensors shard.  Returns the weight_map entries."""
    log.info("Processing %s …", shard_path.name)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(shard_path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    new_tensors: dict[str, torch.Tensor] = {}
    repacked = 0

    for name in sorted(tensors.keys()):
        t = tensors[name]
        if is_moe_bf16_weight(name, t):
            bf16_mb = t.nelement() * t.element_size() / 1e6
            t0 = time.time()
            packed, w_scale, w_scale_2 = quantize_tensor_nvfp4(t)
            elapsed = time.time() - t0
            packed_mb = packed.nelement() * packed.element_size() / 1e6

            base = name  # e.g. "model.layers.3.moe.up_proj.weight"
            new_tensors[base] = packed
            new_tensors[base.replace(".weight", ".weight_scale")] = w_scale
            new_tensors[base.replace(".weight", ".weight_scale_2")] = w_scale_2
            new_tensors[base.replace(".weight", ".input_scale")] = DEFAULT_INPUT_SCALE.clone()

            repacked += 1
            log.info(
                "  %-60s bf16 %.0f MB → uint8 %.0f MB  (%.1fs)",
                name, bf16_mb, packed_mb, elapsed,
            )
        else:
            new_tensors[name] = t

    if repacked == 0:
        log.info("  No MoE bf16 weights in this shard — skipping.")
        return {k: shard_path.name for k in new_tensors}

    save_file(new_tensors, str(shard_path))
    new_size_mb = shard_path.stat().st_size / 1e6
    log.info(
        "  Rewrote %s: %d tensor(s) repacked, new size %.0f MB",
        shard_path.name, repacked, new_size_mb,
    )

    return {k: shard_path.name for k in new_tensors}


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


def main():
    parser = argparse.ArgumentParser(description="Repack bf16 MoE weights to NVFP4")
    parser.add_argument("--ckpt-dir", required=True, help="Path to the NVFP4 checkpoint")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    shards = sorted(ckpt_dir.glob("model-*.safetensors"))
    log.info("Found %d shard(s) in %s", len(shards), ckpt_dir)

    full_weight_map: dict[str, str] = {}
    total_t0 = time.time()

    for shard_path in shards:
        shard_map = repack_shard(shard_path)
        full_weight_map.update(shard_map)

    update_index(ckpt_dir, full_weight_map)

    elapsed = time.time() - total_t0
    log.info("Done in %.0f seconds.", elapsed)

    total_gb = sum(
        s.stat().st_size for s in shards if s.exists()
    ) / 1e9
    log.info("Total checkpoint size: %.1f GB", total_gb)


if __name__ == "__main__":
    main()
