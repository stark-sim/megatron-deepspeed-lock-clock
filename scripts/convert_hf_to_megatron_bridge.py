#!/usr/bin/env python3
"""Convert a Hugging Face model to a Megatron checkpoint using Megatron Bridge.

This helper intentionally loads `megatron.bridge` from a separate Megatron-Bridge
checkout, because this repository already ships a local `megatron/` package that
would otherwise shadow the Bridge package.

Typical usage on a node with 4 GPUs for TP=2, PP=2:

    torchrun --nproc-per-node=4 scripts/convert_hf_to_megatron_bridge.py \
      --bridge-root /path/to/Megatron-Bridge \
      --hf-model /path/to/hf_model \
      --megatron-path /path/to/output_ckpt \
      --tp 2 \
      --pp 2 \
      --torch-dtype bfloat16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _remove_local_repo_from_sys_path() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    filtered = []
    for entry in sys.path:
        if entry in ("", "."):
            continue
        try:
            resolved = Path(entry).resolve()
        except Exception:
            filtered.append(entry)
            continue
        if resolved == repo_root or resolved == repo_root / "scripts":
            continue
        filtered.append(entry)
    sys.path[:] = filtered


def _prepend_bridge_root(bridge_root: Path) -> None:
    bridge_src = bridge_root / "src"
    megatron_lm_root = bridge_root / "3rdparty" / "Megatron-LM"
    if not bridge_src.is_dir():
        raise FileNotFoundError(
            f"MEGATRON_BRIDGE_ROOT is missing src/: {bridge_src}"
        )
    if not megatron_lm_root.is_dir():
        raise FileNotFoundError(
            f"MEGATRON_BRIDGE_ROOT is missing 3rdparty/Megatron-LM: {megatron_lm_root}"
        )
    sys.path.insert(0, str(megatron_lm_root))
    sys.path.insert(0, str(bridge_src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge-root",
        default=os.getenv("MEGATRON_BRIDGE_ROOT", ""),
        help="Path to a Megatron-Bridge checkout. Can also be set via MEGATRON_BRIDGE_ROOT.",
    )
    parser.add_argument("--hf-model", required=True, help="HF model id or local HF model path.")
    parser.add_argument("--megatron-path", required=True, help="Output Megatron checkpoint directory.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--vp", type=int, default=None, help="Virtual pipeline parallel size.")
    parser.add_argument("--cp", type=int, default=None, help="Context parallel size.")
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Torch dtype used while reading HF weights.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _resolve_dtype(dtype_name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def main() -> int:
    args = _parse_args()
    if not args.bridge_root:
        raise SystemExit(
            "MEGATRON_BRIDGE_ROOT is required. Point it at a Megatron-Bridge checkout."
        )

    _remove_local_repo_from_sys_path()
    _prepend_bridge_root(Path(args.bridge_root).resolve())

    import torch
    from megatron.bridge import AutoBridge

    if not AutoBridge.can_handle(args.hf_model, trust_remote_code=args.trust_remote_code):
        raise SystemExit(f"Megatron Bridge cannot handle model: {args.hf_model}")

    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model,
        torch_dtype=_resolve_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )

    # We need explicit provider configuration here because the one-call
    # `import_ckpt()` path from the upstream example does not expose TP/PP.
    provider = bridge.to_megatron_provider(load_weights=False, hf_path=args.hf_model)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = args.pp
    if args.vp is not None:
        provider.virtual_pipeline_model_parallel_size = args.vp
    if args.cp is not None:
        provider.context_parallel_size = args.cp
    provider.finalize()

    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.load_hf_weights(model, hf_path=args.hf_model)

    output_path = Path(args.megatron_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    bridge.save_megatron_model(model, output_path)

    print(f"[Bridge] Saved Megatron checkpoint to: {output_path}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
