#!/usr/bin/env python3
"""
GPU 频率管理器
在 NCCL 通信期间动态调整 GPU 频率以节省能耗

使用 NVML API 进行频率控制，需要 root 权限
"""
import atexit
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from functools import wraps

import torch

# 添加用户 site-packages 路径（sudo 运行时需要）
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("[GPUFreqManager] Warning: pynvml not available")


def get_visible_gpu_indices_from_env() -> Optional[List[int]]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is None or cuda_visible_devices.strip() == "":
        return None

    gpu_indices: List[int] = []
    for item in cuda_visible_devices.split(","):
        token = item.strip()
        if token == "":
            continue
        if not token.isdigit():
            return None
        gpu_indices.append(int(token))
    return gpu_indices if gpu_indices else None


def _decode_nvml_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _try_nvml_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def collect_nvml_device_snapshot(gpu_indices: List[int] = None) -> Dict:
    """Collect a lightweight NVML snapshot for experiment metadata."""
    if not NVML_AVAILABLE:
        return {"available": False, "reason": "pynvml unavailable"}

    if gpu_indices is None:
        gpu_indices = get_visible_gpu_indices_from_env()

    snapshot = {
        "available": True,
        "gpu_count": 0,
        "gpu_indices": [],
        "gpus": [],
    }

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        snapshot["gpu_count"] = device_count

        if gpu_indices is None:
            gpu_indices = list(range(device_count))

        for idx in gpu_indices:
            if idx >= device_count:
                continue

            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            power_limit_mw = _try_nvml_call(
                lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle),
                None,
            )
            default_power_limit_mw = _try_nvml_call(
                lambda: pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle),
                None,
            )
            current_graphics_clock = _try_nvml_call(
                lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
                None,
            )
            current_memory_clock = _try_nvml_call(
                lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                None,
            )
            temperature = _try_nvml_call(
                lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                None,
            )
            mem_clocks = _try_nvml_call(
                lambda: pynvml.nvmlDeviceGetSupportedMemoryClocks(handle),
                [],
            )
            supported_graphics_clocks = []
            if mem_clocks:
                supported_graphics_clocks = _try_nvml_call(
                    lambda: pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mem_clocks[0]),
                    [],
                )

            gpu_info = {
                "index": idx,
                "name": _decode_nvml_name(_try_nvml_call(lambda: pynvml.nvmlDeviceGetName(handle), "unknown")),
                "uuid": _decode_nvml_name(_try_nvml_call(lambda: pynvml.nvmlDeviceGetUUID(handle), None)),
                "temperature_c": temperature,
                "performance_state": _try_nvml_call(lambda: pynvml.nvmlDeviceGetPerformanceState(handle), None),
                "power_limit_w": round(power_limit_mw / 1000.0, 3) if power_limit_mw is not None else None,
                "default_power_limit_w": round(default_power_limit_mw / 1000.0, 3) if default_power_limit_mw is not None else None,
                "current_graphics_clock_mhz": current_graphics_clock,
                "current_memory_clock_mhz": current_memory_clock,
                "supported_memory_clocks_mhz": mem_clocks,
                "supported_graphics_clocks_mhz": supported_graphics_clocks,
            }
            snapshot["gpu_indices"].append(idx)
            snapshot["gpus"].append(gpu_info)
    except Exception as exc:
        snapshot = {"available": False, "reason": str(exc)}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return snapshot


class GPUFreqManager:
    """GPU 频率管理器，用于在通信期间降低 GPU 频率"""
    
    # 频率切换开销约 45ms（16 GPU），只对大于此阈值的通信降频
    # 100M 元素 = 400MB (float32)，在跨节点通信中约需 500-1000ms
    MIN_ELEMENTS_FOR_SCALING = 100 * 1024 * 1024  # 100M elements
    
    # 最小切换间隔（毫秒），避免频繁切换
    MIN_SWITCH_INTERVAL_MS = 200
    
    def __init__(
        self,
        gpu_indices: List[int] = None,
        high_freq: int = None,  # 高频率 (MHz)，None 表示使用默认最高频率
        low_freq: int = 800,    # 低频率 (MHz)，通信期间使用
        enabled: bool = True,
        min_elements: int = None,  # 最小元素数阈值
        dry_run: bool = False,  # dry_run 模式：保留判断逻辑但不执行调频
    ):
        """
        初始化频率管理器
        
        Args:
            gpu_indices: 要管理的 GPU 索引列表，None 表示所有 GPU
            high_freq: 计算时使用的高频率，None 表示使用默认
            low_freq: 通信时使用的低频率
            enabled: 是否启用频率管理
            dry_run: 如果为 True，保留 wrap 函数的判断逻辑但不执行实际调频
        """
        self.enabled = enabled and NVML_AVAILABLE
        self.dry_run = dry_run
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.min_elements = min_elements if min_elements else self.MIN_ELEMENTS_FOR_SCALING
        
        # 统计信息
        self.stats = {
            "total_switches": 0,
            "total_low_freq_time_ms": 0.0,
            "total_high_freq_time_ms": 0.0,
            "comm_count": 0,
            "skipped_small_comm": 0,  # 跳过的小通信次数
        }
        
        self._lock = threading.Lock()
        self._initialized = False
        self._handles: List = []
        self._mem_clock: int = 0
        self._default_graphics_clock: int = 0
        self._current_freq: str = "high"  # "high" or "low"
        self._last_switch_time: float = 0
        self._min_switch_interval = self.MIN_SWITCH_INTERVAL_MS / 1000.0  # 转换为秒
        self._executor: Optional[ThreadPoolExecutor] = None  # 用于并行调频
        
        if not self.enabled:
            print("[GPUFreqManager] Disabled (NVML not available or explicitly disabled)")
            return
        
        try:
            self._init_nvml(gpu_indices)
        except Exception as e:
            print(f"[GPUFreqManager] Failed to initialize: {e}")
            self.enabled = False
    
    def _init_nvml(self, gpu_indices: List[int] = None):
        """初始化 NVML"""
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        if gpu_indices is None:
            gpu_indices = get_visible_gpu_indices_from_env()
        if gpu_indices is None:
            gpu_indices = list(range(device_count))
        
        self._handles = []
        for idx in gpu_indices:
            if idx < device_count:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                self._handles.append((idx, handle))
        
        if not self._handles:
            raise RuntimeError("No valid GPU handles")
        
        # 获取第一个 GPU 的频率信息作为参考
        _, first_handle = self._handles[0]
        
        # 获取内存频率
        mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(first_handle)
        self._mem_clock = mem_clocks[0]  # 使用最高内存频率
        
        # 获取支持的图形频率列表
        supported_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(
            first_handle, self._mem_clock
        )
        
        # 使用最高支持频率作为默认高频（而不是应用时钟）
        self._max_graphics_clock = supported_clocks[0] if supported_clocks else 1597
        
        if self.high_freq is None:
            self.high_freq = self._max_graphics_clock
        
        # 找到最接近目标的支持频率
        self.low_freq = self._find_closest_freq(self.low_freq, supported_clocks)
        self.high_freq = self._find_closest_freq(self.high_freq, supported_clocks)
        
        self._initialized = True
        self._last_switch_time = time.time()
        
        # 初始化线程池用于并行调频
        self._executor = ThreadPoolExecutor(max_workers=len(self._handles))
        
        gpu_name = pynvml.nvmlDeviceGetName(first_handle)
        gpu_indices_str = [idx for idx, _ in self._handles]
        print(f"[GPUFreqManager] Initialized for {len(self._handles)} GPUs ({gpu_name})")
        print(f"[GPUFreqManager] GPU indices: {gpu_indices_str}")
        print(f"[GPUFreqManager] High freq: {self.high_freq} MHz, Low freq: {self.low_freq} MHz")
        print(f"[GPUFreqManager] Memory clock: {self._mem_clock} MHz")
        print(f"[GPUFreqManager] Parallel frequency scaling enabled")
    
    def _find_closest_freq(self, target: int, supported: List[int]) -> int:
        """找到最接近目标的支持频率"""
        if not supported:
            return target
        return min(supported, key=lambda x: abs(x - target))
    
    def set_low_freq(self):
        """设置为低频率（通信期间）"""
        if not self.enabled or not self._initialized:
            return
        
        if self._current_freq == "low":
            return
        
        with self._lock:
            try:
                current_time = time.time()
                
                # 冷却时间检查：避免频繁切换（当前禁用，用于测试阈值）
                # if current_time - self._last_switch_time < self._min_switch_interval:
                #     self.stats["skipped_small_comm"] += 1
                #     return
                
                if self._current_freq == "high":
                    elapsed = (current_time - self._last_switch_time) * 1000
                    self.stats["total_high_freq_time_ms"] += elapsed
                
                # dry_run 模式：只更新统计，不执行实际调频
                if self.dry_run:
                    self._current_freq = "low"
                    self._last_switch_time = current_time
                    self.stats["total_switches"] += 1
                    return
                
                # 并行设置所有 GPU 的频率
                def set_freq_for_gpu(args):
                    idx, handle, freq = args
                    try:
                        pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
                        return (idx, True, None)
                    except pynvml.NVMLError as e:
                        return (idx, False, str(e))
                
                tasks = [(idx, handle, self.low_freq) for idx, handle in self._handles]
                results = list(self._executor.map(set_freq_for_gpu, tasks))
                success_count = sum(1 for _, success, _ in results if success)
                
                if success_count > 0:
                    self._current_freq = "low"
                    self._last_switch_time = current_time
                    self.stats["total_switches"] += 1
                
            except Exception as e:
                print(f"[GPUFreqManager] Unexpected error in set_low_freq: {e}")
    
    def set_high_freq(self):
        """设置为高频率（计算期间）"""
        if not self.enabled or not self._initialized:
            return
        
        if self._current_freq == "high":
            return
        
        with self._lock:
            try:
                current_time = time.time()
                if self._current_freq == "low":
                    elapsed = (current_time - self._last_switch_time) * 1000
                    self.stats["total_low_freq_time_ms"] += elapsed
                
                # dry_run 模式：只更新统计，不执行实际调频
                if self.dry_run:
                    self._current_freq = "high"
                    self._last_switch_time = current_time
                    self.stats["total_switches"] += 1
                    self.stats["comm_count"] += 1
                    return
                
                # 并行设置所有 GPU 的频率
                def set_freq_for_gpu(args):
                    idx, handle, freq = args
                    try:
                        pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
                        return (idx, True, None)
                    except pynvml.NVMLError as e:
                        return (idx, False, str(e))
                
                tasks = [(idx, handle, self.high_freq) for idx, handle in self._handles]
                results = list(self._executor.map(set_freq_for_gpu, tasks))
                success_count = sum(1 for _, success, _ in results if success)
                
                if success_count > 0:
                    self._current_freq = "high"
                    self._last_switch_time = current_time
                    self.stats["total_switches"] += 1
                    self.stats["comm_count"] += 1
                
            except Exception as e:
                print(f"[GPUFreqManager] Unexpected error in set_high_freq: {e}")
    
    def reset_to_default(self):
        """重置为默认频率（完全解锁所有频率限制）"""
        if not self.enabled or not self._initialized:
            return
        
        with self._lock:
            try:
                reset_failures = []
                for idx, handle in self._handles:
                    try:
                        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                    except pynvml.NVMLError as e:
                        reset_failures.append(f"GPU {idx} locked clocks reset failed: {e}")
                    try:
                        pynvml.nvmlDeviceResetApplicationsClocks(handle)
                    except pynvml.NVMLError as e:
                        reset_failures.append(f"GPU {idx} application clocks reset failed: {e}")

                if reset_failures:
                    for message in reset_failures:
                        print(f"[GPUFreqManager] Warning: {message}")

                self._current_freq = "high"
            except Exception as e:
                print(f"[GPUFreqManager] Error in reset_to_default: {e}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._lock:
            # 更新当前状态的时间
            current_time = time.time()
            elapsed = (current_time - self._last_switch_time) * 1000
            
            stats = self.stats.copy()
            if self._current_freq == "high":
                stats["total_high_freq_time_ms"] += elapsed
            else:
                stats["total_low_freq_time_ms"] += elapsed
            
            total_time = stats["total_high_freq_time_ms"] + stats["total_low_freq_time_ms"]
            if total_time > 0:
                stats["low_freq_ratio"] = stats["total_low_freq_time_ms"] / total_time
            else:
                stats["low_freq_ratio"] = 0.0
            
            return stats
    
    def print_stats(self):
        """打印统计信息"""
        global _comm_size_log
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("GPU 频率管理统计")
        print("=" * 60)
        print(f"  阈值: {self.min_elements/1e6:.0f}M 元素")
        print(f"  高频率: {self.high_freq} MHz")
        print(f"  低频率: {self.low_freq} MHz")
        print(f"  总切换次数: {stats['total_switches']}")
        print(f"  通信次数: {stats['comm_count']}")
        print(f"  跳过的小通信: {stats['skipped_small_comm']}")
        print(f"  高频时间: {stats['total_high_freq_time_ms']/1000:.2f} s")
        print(f"  低频时间: {stats['total_low_freq_time_ms']/1000:.2f} s")
        print(f"  低频占比: {stats['low_freq_ratio']*100:.1f}%")
        # 打印通信大小分布
        if _comm_size_log:
            print("  通信大小分布 (MB: 次数):")
            sorted_sizes = sorted(_comm_size_log.items())
            for size_mb, count in sorted_sizes:
                marker = " <-- 触发调频" if size_mb >= self.min_elements/1e6 else ""
                print(f"    {size_mb}M: {count}{marker}")
        print("=" * 60 + "\n")
    
    def shutdown(self):
        """关闭频率管理器"""
        if self._initialized:
            try:
                self.reset_to_default()
                self.print_stats()
            finally:
                if self._executor is not None:
                    self._executor.shutdown(wait=True)
                    self._executor = None
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                self._initialized = False


# 全局频率管理器实例
_freq_manager: Optional[GPUFreqManager] = None
_freq_manager_atexit_registered = False


def get_freq_manager() -> Optional[GPUFreqManager]:
    """获取全局频率管理器"""
    global _freq_manager
    return _freq_manager


def _shutdown_freq_manager_on_exit():
    try:
        shutdown_freq_manager()
    except Exception as exc:
        print(f"[GPUFreqManager] Warning: atexit shutdown failed: {exc}")


def init_freq_manager(
    gpu_indices: List[int] = None,
    high_freq: int = None,
    low_freq: int = 800,
    enabled: bool = True,
    dry_run: bool = False,
    min_elements: int = None,
) -> Optional[GPUFreqManager]:
    """初始化全局频率管理器
    
    Args:
        dry_run: 如果为 True，保留 wrap 函数的判断逻辑但不执行实际调频
        min_elements: 触发调频的最小元素数阈值
    """
    global _freq_manager, _freq_manager_atexit_registered
    _freq_manager = GPUFreqManager(
        gpu_indices=gpu_indices,
        high_freq=high_freq,
        low_freq=low_freq,
        enabled=enabled,
        dry_run=dry_run,
        min_elements=min_elements,
    )
    if not _freq_manager_atexit_registered:
        atexit.register(_shutdown_freq_manager_on_exit)
        _freq_manager_atexit_registered = True
    return _freq_manager


def shutdown_freq_manager():
    """关闭全局频率管理器"""
    global _freq_manager
    if _freq_manager is not None:
        _freq_manager.shutdown()
        _freq_manager = None


# ============================================================
# Monkey-patch torch.distributed 通信函数
# ============================================================

_original_all_reduce = None
_original_all_gather = None
_original_all_gather_base = None
_original_reduce_scatter_base = None
_original_broadcast = None


_comm_size_log = {}  # 记录通信大小分布

def _wrapped_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    """包装的 all_reduce，只对大 tensor 降频"""
    global _freq_manager, _original_all_reduce, _comm_size_log
    
    num_elements = tensor.numel()
    local_rank = os.environ.get('LOCAL_RANK', '?')
    
    # 记录通信大小分布（只在 local_rank 0）
    if local_rank == '0':
        size_mb = int(num_elements / 1e6)
        _comm_size_log[size_mb] = _comm_size_log.get(size_mb, 0) + 1
    
    if _freq_manager is None:
        return _original_all_reduce(tensor, op=op, group=group, async_op=async_op)
    
    if _freq_manager.enabled:
        if num_elements >= _freq_manager.min_elements:
            # 大通信：降频
            if local_rank == '0':
                print(f"[FreqScale] all_reduce {num_elements/1e6:.1f}M >= {_freq_manager.min_elements/1e6:.0f}M, scaling!")
            _freq_manager.set_low_freq()
            try:
                result = _original_all_reduce(tensor, op=op, group=group, async_op=async_op)
                if not async_op:
                    torch.cuda.synchronize()
            finally:
                _freq_manager.set_high_freq()
            return result
        else:
            # 小通信：跳过降频
            _freq_manager.stats["skipped_small_comm"] += 1
            return _original_all_reduce(tensor, op=op, group=group, async_op=async_op)
    else:
        return _original_all_reduce(tensor, op=op, group=group, async_op=async_op)


def _wrapped_all_gather(tensor_list, tensor, group=None, async_op=False):
    """包装的 all_gather，只对大 tensor 降频"""
    global _freq_manager, _original_all_gather
    
    if _freq_manager is not None and _freq_manager.enabled:
        num_elements = tensor.numel() * len(tensor_list)  # 总传输量
        if num_elements >= _freq_manager.min_elements:
            _freq_manager.set_low_freq()
            try:
                result = _original_all_gather(tensor_list, tensor, group=group, async_op=async_op)
                if not async_op:
                    torch.cuda.synchronize()
            finally:
                _freq_manager.set_high_freq()
            return result
        else:
            _freq_manager.stats["skipped_small_comm"] += 1
            return _original_all_gather(tensor_list, tensor, group=group, async_op=async_op)
    else:
        return _original_all_gather(tensor_list, tensor, group=group, async_op=async_op)


def _wrapped_all_gather_base(output, input, group=None, async_op=False):
    """包装的 _all_gather_base，只对大 tensor 降频"""
    global _freq_manager, _original_all_gather_base
    
    if _freq_manager is not None and _freq_manager.enabled:
        num_elements = output.numel()  # output 是收集后的结果
        if num_elements >= _freq_manager.min_elements:
            _freq_manager.set_low_freq()
            try:
                result = _original_all_gather_base(output, input, group=group, async_op=async_op)
                if not async_op:
                    torch.cuda.synchronize()
            finally:
                _freq_manager.set_high_freq()
            return result
        else:
            _freq_manager.stats["skipped_small_comm"] += 1
            return _original_all_gather_base(output, input, group=group, async_op=async_op)
    else:
        return _original_all_gather_base(output, input, group=group, async_op=async_op)


def _wrapped_reduce_scatter_base(output, input, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    """包装的 _reduce_scatter_base，只对大 tensor 降频"""
    global _freq_manager, _original_reduce_scatter_base
    
    if _freq_manager is not None and _freq_manager.enabled:
        num_elements = input.numel()  # input 是 scatter 前的完整数据
        if num_elements >= _freq_manager.min_elements:
            _freq_manager.set_low_freq()
            try:
                result = _original_reduce_scatter_base(output, input, op=op, group=group, async_op=async_op)
                if not async_op:
                    torch.cuda.synchronize()
            finally:
                _freq_manager.set_high_freq()
            return result
        else:
            _freq_manager.stats["skipped_small_comm"] += 1
            return _original_reduce_scatter_base(output, input, op=op, group=group, async_op=async_op)
    else:
        return _original_reduce_scatter_base(output, input, op=op, group=group, async_op=async_op)


def _wrapped_broadcast(tensor, src, group=None, async_op=False):
    """包装的 broadcast，只对大 tensor 降频"""
    global _freq_manager, _original_broadcast
    
    if _freq_manager is not None and _freq_manager.enabled:
        num_elements = tensor.numel()
        if num_elements >= _freq_manager.min_elements:
            _freq_manager.set_low_freq()
            try:
                result = _original_broadcast(tensor, src, group=group, async_op=async_op)
                if not async_op:
                    torch.cuda.synchronize()
            finally:
                _freq_manager.set_high_freq()
            return result
        else:
            _freq_manager.stats["skipped_small_comm"] += 1
            return _original_broadcast(tensor, src, group=group, async_op=async_op)
    else:
        return _original_broadcast(tensor, src, group=group, async_op=async_op)


def patch_torch_distributed():
    """Monkey-patch torch.distributed 通信函数"""
    global _original_all_reduce, _original_all_gather, _original_all_gather_base
    global _original_reduce_scatter_base, _original_broadcast
    
    if _original_all_reduce is not None:
        print("[GPUFreqManager] torch.distributed already patched")
        return
    
    # 保存原始函数
    _original_all_reduce = torch.distributed.all_reduce
    _original_all_gather = torch.distributed.all_gather
    _original_broadcast = torch.distributed.broadcast
    
    if hasattr(torch.distributed, '_all_gather_base'):
        _original_all_gather_base = torch.distributed._all_gather_base
    if hasattr(torch.distributed, '_reduce_scatter_base'):
        _original_reduce_scatter_base = torch.distributed._reduce_scatter_base
    
    # 替换为包装函数
    torch.distributed.all_reduce = _wrapped_all_reduce
    torch.distributed.all_gather = _wrapped_all_gather
    torch.distributed.broadcast = _wrapped_broadcast
    
    if _original_all_gather_base is not None:
        torch.distributed._all_gather_base = _wrapped_all_gather_base
    if _original_reduce_scatter_base is not None:
        torch.distributed._reduce_scatter_base = _wrapped_reduce_scatter_base
    
    print("[GPUFreqManager] torch.distributed patched for frequency scaling")


def unpatch_torch_distributed():
    """恢复 torch.distributed 原始函数"""
    global _original_all_reduce, _original_all_gather, _original_all_gather_base
    global _original_reduce_scatter_base, _original_broadcast
    
    if _original_all_reduce is None:
        return
    
    torch.distributed.all_reduce = _original_all_reduce
    torch.distributed.all_gather = _original_all_gather
    torch.distributed.broadcast = _original_broadcast
    
    if _original_all_gather_base is not None:
        torch.distributed._all_gather_base = _original_all_gather_base
    if _original_reduce_scatter_base is not None:
        torch.distributed._reduce_scatter_base = _original_reduce_scatter_base
    
    _original_all_reduce = None
    _original_all_gather = None
    _original_all_gather_base = None
    _original_reduce_scatter_base = None
    _original_broadcast = None
    
    print("[GPUFreqManager] torch.distributed unpatched")


def enable_comm_freq_scaling(
    gpu_indices: List[int] = None,
    high_freq: int = None,
    low_freq: int = 800,
):
    """
    启用通信期间的频率缩放
    
    Args:
        gpu_indices: 要管理的 GPU 索引
        high_freq: 计算时的高频率 (MHz)
        low_freq: 通信时的低频率 (MHz)
    """
    manager = init_freq_manager(
        gpu_indices=gpu_indices,
        high_freq=high_freq,
        low_freq=low_freq,
        enabled=True,
    )
    
    if manager is not None and manager.enabled:
        patch_torch_distributed()
        return True
    return False


def disable_comm_freq_scaling():
    """禁用通信期间的频率缩放"""
    unpatch_torch_distributed()
    shutdown_freq_manager()


if __name__ == '__main__':
    # 测试频率管理器
    print("测试 GPU 频率管理器...")
    
    manager = GPUFreqManager(low_freq=800)
    
    if manager.enabled:
        print("\n测试频率切换...")
        
        print("设置低频...")
        manager.set_low_freq()
        time.sleep(1)
        
        print("设置高频...")
        manager.set_high_freq()
        time.sleep(1)
        
        manager.shutdown()
    else:
        print("频率管理器未启用（可能需要 root 权限）")
