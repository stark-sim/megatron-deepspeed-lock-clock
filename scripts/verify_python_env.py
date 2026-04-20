#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from typing import Any, Dict


def module_version(module: Any) -> str:
    return getattr(module, "__version__", "unknown")


def check_import(module_name: str, attr_name: str | None = None) -> Dict[str, Any]:
    try:
        module = __import__(module_name, fromlist=["*"])
        payload: Dict[str, Any] = {
            "ok": True,
            "module": module_name,
            "version": module_version(module),
        }
        if attr_name:
            payload["attr_present"] = hasattr(module, attr_name)
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "module": module_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def warmup_apex(torch_module: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True}
    try:
        from apex.optimizers import FusedAdam
        from apex.normalization import MixedFusedRMSNorm
        from apex.contrib.layer_norm.layer_norm import FastLayerNormFN  # noqa: F401

        result["fused_adam_import"] = True
        result["rmsnorm_import"] = True

        if torch_module.cuda.is_available():
            param = torch_module.nn.Parameter(torch_module.zeros(16, device="cuda"))
            optimizer = FusedAdam([param], lr=1e-3)
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            layer = MixedFusedRMSNorm(16).cuda()
            _ = layer(torch_module.randn(2, 16, device="cuda"))
            result["cuda_step"] = True
        else:
            result["cuda_step"] = False
            result["note"] = "cuda unavailable, import-only apex validation"
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def warmup_deepspeed(torch_module: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True}
    try:
        from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam as DSFusedAdam

        cpu_param = torch_module.nn.Parameter(torch_module.zeros(16))
        cpu_optimizer = DeepSpeedCPUAdam([cpu_param], lr=1e-3)
        cpu_param.grad = torch_module.zeros_like(cpu_param)
        cpu_optimizer.step()
        cpu_optimizer.zero_grad()
        result["cpu_adam_step"] = True

        if torch_module.cuda.is_available():
            gpu_param = torch_module.nn.Parameter(torch_module.zeros(16, device="cuda"))
            gpu_optimizer = DSFusedAdam([gpu_param], lr=1e-3)
            gpu_loss = (gpu_param ** 2).sum()
            gpu_loss.backward()
            gpu_optimizer.step()
            gpu_optimizer.zero_grad(set_to_none=True)
            result["fused_adam_step"] = True
        else:
            result["fused_adam_step"] = False
            result["note"] = "cuda unavailable, skipped DS fused adam step"
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def warmup_megatron_imports() -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True}
    try:
        import megatron  # noqa: F401
        import pretrain_gpt  # noqa: F401

        result["imports"] = ["megatron", "pretrain_gpt"]
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


def collect_env_snapshot() -> Dict[str, Any]:
    interesting = [
        "PYTHONPATH",
        "TORCH_EXTENSIONS_DIR",
        "TMPDIR",
        "PYTHONPYCACHEPREFIX",
        "TRITON_CACHE_DIR",
        "PYTHONDONTWRITEBYTECODE",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_NCCL_BLOCKING_WAIT",
        "MAX_JOBS",
    ]
    return {key: os.environ.get(key, "") for key in interesting}


def query_nvcc() -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "stdout_tail": "\n".join(completed.stdout.strip().splitlines()[-2:]),
        }
    except FileNotFoundError:
        return {"ok": False, "stdout_tail": "nvcc not found"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Python runtime for Megatron-DeepSpeed lock-clock experiments.")
    parser.add_argument("--warmup", action="store_true", help="Run DeepSpeed/Apex warmup steps, not just imports.")
    parser.add_argument("--json-output", type=str, default="", help="Optional path to save the JSON report.")
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env": collect_env_snapshot(),
        "nvcc": query_nvcc(),
        "checks": {},
    }

    torch_check = check_import("torch")
    report["checks"]["torch"] = torch_check
    if not torch_check["ok"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    import torch

    report["checks"]["deepspeed"] = check_import("deepspeed")
    report["checks"]["transformers"] = check_import("transformers")
    report["checks"]["einops"] = check_import("einops")
    report["checks"]["sentencepiece"] = check_import("sentencepiece")
    report["checks"]["pynvml"] = check_import("pynvml")
    report["checks"]["zeus"] = check_import("zeus")
    report["checks"]["apex"] = check_import("apex")

    report["cuda"] = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "torch_version": getattr(torch, "__version__", "unknown"),
    }
    if torch.cuda.is_available():
        report["cuda"]["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    if args.warmup:
        report["warmup"] = {
            "apex": warmup_apex(torch),
            "deepspeed": warmup_deepspeed(torch),
            "megatron": warmup_megatron_imports(),
        }

    failed = [
        name for name, payload in report["checks"].items()
        if not payload.get("ok", False)
    ]
    if args.warmup:
        failed.extend(
            name for name, payload in report["warmup"].items()
            if not payload.get("ok", False)
        )
    report["ok"] = len(failed) == 0
    report["failed"] = failed

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
