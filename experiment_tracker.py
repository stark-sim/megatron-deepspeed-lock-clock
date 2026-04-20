#!/usr/bin/env python3

import atexit
import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from megatron.gpu_freq_manager import collect_nvml_device_snapshot


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _run_command(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def _pick_args(args, names):
    return {
        name: _json_safe(getattr(args, name))
        for name in names
        if hasattr(args, name)
    }


def _read_json_file_from_env(env_name: str) -> Optional[Dict[str, Any]]:
    path = os.getenv(env_name)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def build_frequency_scaling_snapshot(manager) -> Optional[Dict[str, Any]]:
    if manager is None:
        return None

    return {
        "enabled": bool(getattr(manager, "enabled", False)),
        "dry_run": bool(getattr(manager, "dry_run", False)),
        "high_freq_mhz": getattr(manager, "high_freq", None),
        "low_freq_mhz": getattr(manager, "low_freq", None),
        "min_elements": getattr(manager, "min_elements", None),
        "current_freq_state": getattr(manager, "_current_freq", None),
        "stats": _json_safe(manager.get_stats()),
    }


class ExperimentTracker:
    def __init__(self, run_id: str, experiment_name: str, root_dir: str):
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.root_dir = os.path.abspath(root_dir)
        self.run_dir = os.path.join(self.root_dir, self.run_id)
        self.run_file = os.path.join(self.run_dir, "run.json")
        self.events_file = os.path.join(self.run_dir, "events.jsonl")
        self.notes_file = os.path.join(self.run_dir, "notes.md")
        self.command_file = os.path.join(self.run_dir, "command.sh")
        self.index_file = os.path.join(self.root_dir, "index.jsonl")
        self.finalized = False
        atexit.register(self._finalize_on_exit)

    def initialize(self, args) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        command = os.getenv("MEGATRON_LAUNCH_COMMAND") or shlex.join([sys.executable, *sys.argv])
        command_hash = hashlib.sha1(command.encode("utf-8")).hexdigest()
        manifest = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": "initialized",
            "created_at": _utc_now(),
            "identity": {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
            },
            "command": {
                "argv": sys.argv,
                "command": command,
                "sha1": command_hash,
                "launcher_script": os.getenv("MEGATRON_LAUNCHER_SCRIPT"),
            },
            "paths": {
                "root_dir": self.root_dir,
                "run_dir": self.run_dir,
                "run_file": self.run_file,
                "events_file": self.events_file,
                "notes_file": self.notes_file,
                "command_file": self.command_file,
                "save_dir": getattr(args, "save", None),
                "load_dir": getattr(args, "load", None),
            },
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "cwd": os.getcwd(),
                "git_commit": _run_command("git", "rev-parse", "HEAD"),
                "git_branch": _run_command("git", "branch", "--show-current"),
            },
            "environment": {
                key: os.getenv(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "WORLD_SIZE",
                    "RANK",
                    "LOCAL_RANK",
                    "NODE_RANK",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "HOSTNAME",
                    "MEGATRON_RUN_ID",
                    "MEGATRON_EXPERIMENT_ROOT",
                    "MEGATRON_EXPERIMENT_MODE",
                    "MEGATRON_HOSTFILE_PATH",
                ]
                if os.getenv(key) is not None
            },
            "topology": {
                "requested": {
                    "tp": os.getenv("MEGATRON_REQUESTED_TP"),
                    "pp": os.getenv("MEGATRON_REQUESTED_PP"),
                },
                "resolved": _read_json_file_from_env("MEGATRON_TOPOLOGY_JSON") or {},
            },
            "hostfile": _read_json_file_from_env("MEGATRON_HOSTFILE_JSON") or {},
            "preflight": _read_json_file_from_env("MEGATRON_PREFLIGHT_JSON") or {},
            "config": {
                "model": _pick_args(args, [
                    "num_layers",
                    "hidden_size",
                    "ffn_hidden_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "seq_length",
                    "max_position_embeddings",
                    "tokenizer_type",
                    "tokenizer_model",
                    "normalization",
                    "swiglu",
                    "use_rotary_position_embeddings",
                ]),
                "training": _pick_args(args, [
                    "micro_batch_size",
                    "global_batch_size",
                    "train_iters",
                    "lr",
                    "min_lr",
                    "lr_decay_style",
                    "lr_warmup_iters",
                    "weight_decay",
                    "clip_grad",
                    "bf16",
                    "fp16",
                    "save_interval",
                    "eval_interval",
                    "eval_iters",
                ]),
                "parallelism": _pick_args(args, [
                    "tensor_model_parallel_size",
                    "pipeline_model_parallel_size",
                    "sequence_parallel",
                    "zero_stage",
                    "DDP_impl",
                ]),
                "data": _pick_args(args, [
                    "data_path",
                    "split",
                    "data_impl",
                ]),
                "frequency_scaling": _pick_args(args, [
                    "enable_comm_freq_scaling",
                    "comm_low_freq",
                    "comm_high_freq",
                    "comm_freq_dry_run",
                    "comm_min_elements",
                ]),
            },
            "freq_policy": {
                "mode": os.getenv("MEGATRON_EXPERIMENT_MODE"),
                "static_clock_mhz": os.getenv("STATIC_CLOCK_MHZ"),
                "comm_low_freq": os.getenv("COMM_LOW_FREQ"),
                "comm_high_freq": os.getenv("COMM_HIGH_FREQ"),
                "comm_min_elements": os.getenv("COMM_MIN_ELEMENTS"),
            },
            "nvml": _json_safe(collect_nvml_device_snapshot()),
        }

        self._write_command_file(command)
        self._write_notes_template(manifest)
        _write_json(self.run_file, manifest)
        self.append_event("initialized", {
            "status": "initialized",
            "command_sha1": command_hash,
        })
        self._append_index({
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": "initialized",
            "command_sha1": command_hash,
            "run_dir": self.run_dir,
        })

    def append_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        entry = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "event_type": event_type,
            "payload": _json_safe(payload or {}),
        }
        with open(self.events_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")

    def update_manifest(self, update: Dict[str, Any]) -> None:
        current = _read_json(self.run_file)
        merged = _deep_merge(current, _json_safe(update))
        _write_json(self.run_file, merged)

    def record_interval(self, iteration: int, interval_metrics: Dict[str, Any], freq_metrics: Optional[Dict[str, Any]]) -> None:
        payload = {
            "iteration": iteration,
            "interval_metrics": interval_metrics,
            "frequency_scaling": freq_metrics,
        }
        self.append_event("interval", payload)

    def record_checkpoint(self, iteration: int, checkpoint_dir: Optional[str], extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "iteration": iteration,
            "checkpoint_dir": checkpoint_dir,
        }
        if extra:
            payload.update(extra)
        self.append_event("checkpoint", payload)

    def finalize(self, status: str, final_iteration: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        if self.finalized:
            return
        update = {
            "status": status,
            "ended_at": _utc_now(),
        }
        if final_iteration is not None:
            update["final_iteration"] = final_iteration
        if extra:
            update.update(extra)
        self.update_manifest(update)
        self.append_event("finalized", {
            "status": status,
            "final_iteration": final_iteration,
            "extra": extra or {},
        })
        self._append_index({
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": status,
            "run_dir": self.run_dir,
        })
        self.finalized = True

    def _append_index(self, payload: Dict[str, Any]) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        with open(self.index_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True))
            handle.write("\n")

    def _write_command_file(self, command: str) -> None:
        if os.path.exists(self.command_file):
            return
        with open(self.command_file, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\n")
            handle.write(f"export MEGATRON_RUN_ID={shlex.quote(self.run_id)}\n")
            handle.write(f"export MEGATRON_EXPERIMENT_ROOT={shlex.quote(self.root_dir)}\n")
            handle.write(command)
            handle.write("\n")
        os.chmod(self.command_file, 0o755)

    def _write_notes_template(self, manifest: Dict[str, Any]) -> None:
        if os.path.exists(self.notes_file):
            return
        command = manifest["command"]["command"]
        lines = [
            f"# Experiment {self.run_id}",
            "",
            "## Metadata",
            f"- experiment_name: `{self.experiment_name}`",
            f"- status: `{manifest['status']}`",
            f"- run_dir: `{self.run_dir}`",
            f"- command_sha1: `{manifest['command']['sha1']}`",
            "",
            "## Command",
            "```bash",
            command,
            "```",
            "",
            "## Hypothesis",
            "- ",
            "",
            "## Setup Notes",
            "- GPU / node:",
            "- cooling / ambient:",
            "- driver / CUDA:",
            "- clock policy:",
            "",
            "## Key Metrics",
            "- step time:",
            "- tokens per second:",
            "- average power:",
            "- energy per 50 steps:",
            "- loss curve note:",
            "",
            "## Comparison",
            "- baseline run_id:",
            "- compared dimension:",
            "- conclusion:",
        ]
        with open(self.notes_file, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _finalize_on_exit(self) -> None:
        if not self.finalized and os.path.exists(self.run_file):
            try:
                self.update_manifest({
                    "status": "incomplete",
                    "ended_at": _utc_now(),
                })
            except Exception:
                pass


_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker() -> Optional[ExperimentTracker]:
    return _tracker


def init_experiment_tracker(args) -> Optional[ExperimentTracker]:
    global _tracker
    run_id = getattr(args, "experiment_run_id", None) or os.getenv("MEGATRON_RUN_ID")
    if not run_id:
        return None

    experiment_name = getattr(args, "experiment_name", None) or run_id
    root_dir = (
        getattr(args, "experiment_root_dir", None)
        or os.getenv("MEGATRON_EXPERIMENT_ROOT")
        or os.path.join(os.getcwd(), "experiments")
    )
    _tracker = ExperimentTracker(run_id=run_id, experiment_name=experiment_name, root_dir=root_dir)
    _tracker.initialize(args)
    return _tracker


def record_experiment_interval(iteration: int, interval_metrics: Dict[str, Any], freq_metrics: Optional[Dict[str, Any]] = None) -> None:
    if _tracker is not None:
        _tracker.record_interval(iteration, interval_metrics, freq_metrics)


def record_experiment_checkpoint(iteration: int, checkpoint_dir: Optional[str], extra: Optional[Dict[str, Any]] = None) -> None:
    if _tracker is not None:
        _tracker.record_checkpoint(iteration, checkpoint_dir, extra=extra)


def finalize_experiment_tracker(status: str, final_iteration: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    if _tracker is not None:
        _tracker.finalize(status=status, final_iteration=final_iteration, extra=extra)
