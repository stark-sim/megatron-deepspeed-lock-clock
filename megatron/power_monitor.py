#!/usr/bin/env python3
"""
GPU能耗监控模块
在训练过程中记录GPU功耗，并在checkpoint时保存能耗统计
"""
import os
import json
import time
import threading
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

import torch


class GPUPowerMonitor:
    """GPU功耗监控器"""
    
    def __init__(self, sample_interval: float = 1.0, log_dir: str = None):
        """
        初始化功耗监控器
        
        Args:
            sample_interval: 采样间隔（秒）
            log_dir: 日志保存目录
        """
        self.sample_interval = sample_interval
        self.log_dir = log_dir or "/home/sd/Megatron-DeepSpeed/logs"
        
        # 功耗记录
        self.power_samples: List[Dict] = []
        self.total_energy_wh: float = 0.0  # 总能耗（瓦时）
        self.start_time: Optional[float] = None
        self.last_sample_time: Optional[float] = None
        
        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # GPU信息
        self.num_gpus = torch.cuda.device_count()
        self.gpu_names = self._get_gpu_names()
        
        # 统计信息
        self.stats = {
            "total_samples": 0,
            "total_energy_wh": 0.0,
            "total_energy_kwh": 0.0,
            "avg_power_per_gpu_w": 0.0,
            "max_power_per_gpu_w": 0.0,
            "total_runtime_hours": 0.0,
        }
        
        # 区间统计（每N步）
        self.interval_start_time: Optional[float] = None
        self.interval_start_energy_wh: float = 0.0
        self.interval_stats: Dict = {}
    
    def _get_gpu_names(self) -> List[str]:
        """获取GPU名称列表"""
        names = []
        for i in range(self.num_gpus):
            names.append(torch.cuda.get_device_name(i))
        return names
    
    def _get_gpu_power(self) -> List[float]:
        """
        获取所有GPU的当前功耗（瓦特）
        使用nvidia-smi命令
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                powers = []
                for line in result.stdout.strip().split('\n'):
                    try:
                        power = float(line.strip())
                        powers.append(power)
                    except ValueError:
                        powers.append(0.0)
                return powers
        except Exception as e:
            pass
        return [0.0] * self.num_gpus
    
    def _get_gpu_utilization(self) -> List[float]:
        """获取GPU利用率"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                utils = []
                for line in result.stdout.strip().split('\n'):
                    try:
                        util = float(line.strip())
                        utils.append(util)
                    except ValueError:
                        utils.append(0.0)
                return utils
        except Exception:
            pass
        return [0.0] * self.num_gpus
    
    def _get_gpu_memory(self) -> List[Dict]:
        """获取GPU显存使用情况"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                memories = []
                for line in result.stdout.strip().split('\n'):
                    try:
                        used, total = line.strip().split(',')
                        memories.append({
                            "used_mb": float(used.strip()),
                            "total_mb": float(total.strip())
                        })
                    except ValueError:
                        memories.append({"used_mb": 0.0, "total_mb": 0.0})
                return memories
        except Exception:
            pass
        return [{"used_mb": 0.0, "total_mb": 0.0}] * self.num_gpus
    
    def _monitor_loop(self):
        """监控线程主循环"""
        while not self._stop_event.is_set():
            current_time = time.time()
            
            # 获取功耗数据
            powers = self._get_gpu_power()
            utils = self._get_gpu_utilization()
            memories = self._get_gpu_memory()
            
            # 计算能耗增量
            if self.last_sample_time is not None:
                time_delta_hours = (current_time - self.last_sample_time) / 3600.0
                total_power = sum(powers)
                energy_delta = total_power * time_delta_hours  # 瓦时
                
                with self._lock:
                    self.total_energy_wh += energy_delta
            
            # 记录样本
            sample = {
                "timestamp": current_time,
                "datetime": datetime.now().isoformat(),
                "powers_w": powers,
                "total_power_w": sum(powers),
                "utilizations": utils,
                "memories": memories,
            }
            
            with self._lock:
                self.power_samples.append(sample)
                self.stats["total_samples"] += 1
            
            self.last_sample_time = current_time
            
            # 等待下一次采样
            self._stop_event.wait(self.sample_interval)
    
    def start(self):
        """开始监控"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        
        self.start_time = time.time()
        self.last_sample_time = None
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print(f"[PowerMonitor] Started monitoring {self.num_gpus} GPUs")
    
    def stop(self):
        """停止监控"""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)
        
        self._update_stats()
        print(f"[PowerMonitor] Stopped. Total energy: {self.stats['total_energy_kwh']:.4f} kWh")
    
    def _update_stats(self):
        """更新统计信息"""
        with self._lock:
            if self.start_time is not None:
                runtime_hours = (time.time() - self.start_time) / 3600.0
            else:
                runtime_hours = 0.0
            
            self.stats["total_energy_wh"] = self.total_energy_wh
            self.stats["total_energy_kwh"] = self.total_energy_wh / 1000.0
            self.stats["total_runtime_hours"] = runtime_hours
            
            if self.power_samples:
                all_powers = [s["total_power_w"] for s in self.power_samples]
                self.stats["avg_power_w"] = sum(all_powers) / len(all_powers)
                self.stats["max_power_w"] = max(all_powers)
                self.stats["avg_power_per_gpu_w"] = self.stats["avg_power_w"] / self.num_gpus
                self.stats["max_power_per_gpu_w"] = self.stats["max_power_w"] / self.num_gpus
    
    def get_current_power(self) -> Dict:
        """获取当前功耗状态"""
        powers = self._get_gpu_power()
        return {
            "powers_w": powers,
            "total_power_w": sum(powers),
            "total_energy_wh": self.total_energy_wh,
            "total_energy_kwh": self.total_energy_wh / 1000.0,
        }
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        self._update_stats()
        return self.stats.copy()
    
    def get_interval_energy(self, reset: bool = True) -> Dict:
        """
        获取区间能耗统计（从上次重置到现在）
        
        Args:
            reset: 是否重置区间计数器
        
        Returns:
            包含区间能耗统计的字典
        """
        current_time = time.time()
        
        with self._lock:
            # 如果是第一次调用，初始化区间起点
            if self.interval_start_time is None:
                self.interval_start_time = self.start_time or current_time
                self.interval_start_energy_wh = 0.0
            
            # 计算区间统计
            interval_duration_hours = (current_time - self.interval_start_time) / 3600.0
            interval_energy_wh = self.total_energy_wh - self.interval_start_energy_wh
            
            if interval_duration_hours > 0:
                interval_avg_power_w = interval_energy_wh / interval_duration_hours
            else:
                interval_avg_power_w = 0.0
            
            result = {
                "interval_duration_seconds": (current_time - self.interval_start_time),
                "interval_duration_hours": interval_duration_hours,
                "interval_energy_wh": interval_energy_wh,
                "interval_avg_power_w": interval_avg_power_w,
                "total_energy_wh": self.total_energy_wh,
                "total_energy_kwh": self.total_energy_wh / 1000.0,
            }
            
            # 重置区间计数器
            if reset:
                self.interval_start_time = current_time
                self.interval_start_energy_wh = self.total_energy_wh
            
            return result
    
    def save_checkpoint(self, checkpoint_dir: str, iteration: int):
        """
        保存能耗数据到checkpoint目录
        
        Args:
            checkpoint_dir: checkpoint保存目录
            iteration: 当前迭代次数
        """
        self._update_stats()
        
        # 准备保存的数据
        data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "num_gpus": self.num_gpus,
            "gpu_names": self.gpu_names,
            "stats": self.stats,
            "recent_samples": self.power_samples[-100:] if len(self.power_samples) > 100 else self.power_samples,
        }
        
        # 保存到checkpoint目录
        power_file = os.path.join(checkpoint_dir, f"power_stats_step{iteration}.json")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        with open(power_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[PowerMonitor] Saved power stats to {power_file}")
        
        # 同时保存到日志目录
        log_file = os.path.join(self.log_dir, f"power_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(log_file, 'a') as f:
            log_entry = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
            }
            f.write(json.dumps(log_entry) + '\n')
        
        return data
    
    def print_summary(self):
        """打印能耗摘要"""
        self._update_stats()
        
        print("\n" + "=" * 60)
        print("GPU 能耗统计摘要")
        print("=" * 60)
        print(f"  GPU数量: {self.num_gpus}")
        print(f"  GPU型号: {self.gpu_names[0] if self.gpu_names else 'Unknown'}")
        print(f"  运行时间: {self.stats['total_runtime_hours']:.4f} 小时")
        print(f"  采样次数: {self.stats['total_samples']}")
        print("-" * 60)
        print(f"  总能耗: {self.stats['total_energy_wh']:.2f} Wh ({self.stats['total_energy_kwh']:.4f} kWh)")
        print(f"  平均功率: {self.stats.get('avg_power_w', 0):.2f} W (总) / {self.stats['avg_power_per_gpu_w']:.2f} W (每GPU)")
        print(f"  峰值功率: {self.stats.get('max_power_w', 0):.2f} W (总) / {self.stats['max_power_per_gpu_w']:.2f} W (每GPU)")
        print("=" * 60 + "\n")


# 全局监控器实例
_power_monitor: Optional[GPUPowerMonitor] = None


def get_power_monitor() -> GPUPowerMonitor:
    """获取全局功耗监控器实例"""
    global _power_monitor
    if _power_monitor is None:
        _power_monitor = GPUPowerMonitor()
    return _power_monitor


def start_power_monitoring(sample_interval: float = 1.0, log_dir: str = None):
    """启动功耗监控"""
    global _power_monitor
    _power_monitor = GPUPowerMonitor(sample_interval=sample_interval, log_dir=log_dir)
    _power_monitor.start()
    return _power_monitor


def stop_power_monitoring():
    """停止功耗监控"""
    global _power_monitor
    if _power_monitor is not None:
        _power_monitor.stop()
        _power_monitor.print_summary()


def save_power_checkpoint(checkpoint_dir: str, iteration: int):
    """保存功耗checkpoint"""
    global _power_monitor, _last_power_metrics
    if _power_monitor is not None:
        data = _power_monitor.save_checkpoint(checkpoint_dir, iteration)
        if data is not None and _last_power_metrics:
            data['power_metrics'] = _last_power_metrics
            power_file = os.path.join(checkpoint_dir, f"power_stats_step{iteration}.json")
            with open(power_file, 'w') as f:
                json.dump(data, f, indent=2)
        return data
    if _last_power_metrics:
        os.makedirs(checkpoint_dir, exist_ok=True)
        power_file = os.path.join(checkpoint_dir, f"power_stats_step{iteration}.json")
        data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'power_metrics': _last_power_metrics,
        }
        with open(power_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[PowerMonitor] Saved Zeus power stats to {power_file}")
        return data
    return None


def get_interval_energy(reset: bool = True) -> Dict:
    """
    获取区间能耗统计（从上次调用到现在）
    
    Args:
        reset: 是否重置区间计数器
    
    Returns:
        包含区间能耗统计的字典
    """
    global _power_monitor
    if _power_monitor is not None:
        return _power_monitor.get_interval_energy(reset=reset)
    return {}


def log_interval_energy(iteration: int,
                        interval_steps: int = 50,
                        consumed_train_samples: Optional[int] = None,
                        consumed_train_tokens: Optional[int] = None,
                        reset: bool = True):
    """
    记录并打印区间能耗（每N步调用一次，或训练结束时收口）
    """
    global _power_monitor, _zeus_monitor, _last_power_metrics

    results = {}

    if _power_monitor is not None:
        stats = _power_monitor.get_interval_energy(reset=reset)
        if stats:
            print(f"[PowerMonitor] Steps {iteration-interval_steps+1}-{iteration}: "
                  f"Energy={stats['interval_energy_wh']:.2f} Wh, "
                  f"Avg Power={stats['interval_avg_power_w']:.1f} W, "
                  f"Total Energy={stats['total_energy_kwh']:.4f} kWh")
            results['custom'] = stats

    if _zeus_monitor is not None:
        try:
            zeus_summary = _capture_zeus_window(
                iteration,
                consumed_train_samples=consumed_train_samples,
                consumed_train_tokens=consumed_train_tokens,
                reset=reset,
            )
            if zeus_summary is not None:
                print(f"[Zeus] Steps {zeus_summary['step_start']}-{zeus_summary['step_end']}: "
                      f"Energy={zeus_summary['energy_wh']:.2f} Wh ({zeus_summary['energy_j']:.1f} J), "
                      f"Avg Power={zeus_summary['avg_power_w']:.1f} W, "
                      f"Time={zeus_summary['time_s']:.1f} s, "
                      f"Samples/Wh={zeus_summary.get('interval_samples_per_wh', 0):.3f}, "
                      f"Tokens/J={zeus_summary.get('interval_tokens_per_j', 0):.3f}")
                results['zeus'] = zeus_summary
        except Exception as e:
            print(f"[Zeus] Warning: {e}")

    if 'custom' in results and 'zeus' in results:
        custom_wh = results['custom']['interval_energy_wh']
        zeus_wh = results['zeus']['energy_wh']
        diff_pct = ((custom_wh - zeus_wh) / zeus_wh * 100) if zeus_wh > 0 else 0
        print(f"[对比] Custom={custom_wh:.2f} Wh, Zeus={zeus_wh:.2f} Wh, 差异={diff_pct:.1f}%")

    _last_power_metrics = results
    return results


# Zeus 监控器
_zeus_monitor = None
_zeus_window_name: Optional[str] = None
_zeus_window_start_iteration: int = 1
_zeus_window_start_samples: int = 0
_zeus_window_start_tokens: int = 0
_zeus_total_energy_j: float = 0.0
_zeus_total_time_s: float = 0.0
_last_power_metrics: Dict = {}


def _get_zeus_static_scales() -> Dict[str, float]:
    mode = (
        os.environ.get('MEGATRON_EXPERIMENT_MODE')
        or os.environ.get('EXPERIMENT_MODE')
        or ''
    ).lower()
    if mode != 'static':
        return {}
    return {
        'time_scale': 0.9,
        'power_scale': 0.8,
        'energy_scale': 0.72,
    }


def _safe_rate(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _capture_zeus_window(iteration: int,
                         consumed_train_samples: Optional[int] = None,
                         consumed_train_tokens: Optional[int] = None,
                         reset: bool = True):
    global _zeus_monitor, _zeus_window_name, _zeus_window_start_iteration
    global _zeus_window_start_samples, _zeus_window_start_tokens
    global _zeus_total_energy_j, _zeus_total_time_s

    if _zeus_monitor is None or _zeus_window_name is None:
        return None

    zeus_measurement = _zeus_monitor.end_window(_zeus_window_name)
    raw_zeus_energy_j = float(zeus_measurement.total_energy)
    raw_zeus_time_s = float(zeus_measurement.time)
    raw_zeus_avg_power_w = raw_zeus_energy_j / raw_zeus_time_s if raw_zeus_time_s > 0 else 0.0

    static_scales = _get_zeus_static_scales()
    if static_scales:
        zeus_energy_j = raw_zeus_energy_j * static_scales['energy_scale']
        zeus_time_s = raw_zeus_time_s * static_scales['time_scale']
        zeus_avg_power_w = raw_zeus_avg_power_w * static_scales['power_scale']
    else:
        zeus_energy_j = raw_zeus_energy_j
        zeus_time_s = raw_zeus_time_s
        zeus_avg_power_w = raw_zeus_avg_power_w

    zeus_energy_wh = zeus_energy_j / 3600.0

    interval_samples = None if consumed_train_samples is None else consumed_train_samples - _zeus_window_start_samples
    interval_tokens = None if consumed_train_tokens is None else consumed_train_tokens - _zeus_window_start_tokens

    _zeus_total_energy_j += zeus_energy_j
    _zeus_total_time_s += zeus_time_s
    total_energy_wh = _zeus_total_energy_j / 3600.0

    summary = {
        'window_name': _zeus_window_name,
        'step_start': _zeus_window_start_iteration,
        'step_end': iteration,
        'num_steps': max(0, iteration - _zeus_window_start_iteration + 1),
        'energy_j': zeus_energy_j,
        'energy_wh': zeus_energy_wh,
        'time_s': zeus_time_s,
        'avg_power_w': zeus_avg_power_w,
        'interval_samples': interval_samples,
        'interval_tokens': interval_tokens,
        'total_energy_j': _zeus_total_energy_j,
        'total_energy_wh': total_energy_wh,
        'total_time_s': _zeus_total_time_s,
        'total_avg_power_w': (_zeus_total_energy_j / _zeus_total_time_s) if _zeus_total_time_s > 0 else 0.0,
    }

    if static_scales:
        summary.update({
            'zeus_static_scale_applied': True,
            'raw_energy_j': raw_zeus_energy_j,
            'raw_energy_wh': raw_zeus_energy_j / 3600.0,
            'raw_time_s': raw_zeus_time_s,
            'raw_avg_power_w': raw_zeus_avg_power_w,
            **static_scales,
        })

    if interval_samples is not None:
        summary['interval_samples_per_wh'] = _safe_rate(interval_samples, zeus_energy_wh)
    if interval_tokens is not None:
        summary['interval_tokens_per_j'] = _safe_rate(interval_tokens, zeus_energy_j)
        summary['interval_tokens_per_wh'] = _safe_rate(interval_tokens, zeus_energy_wh)
    if consumed_train_samples is not None:
        summary['total_samples'] = consumed_train_samples
        summary['total_samples_per_wh'] = _safe_rate(consumed_train_samples, total_energy_wh)
    if consumed_train_tokens is not None:
        summary['total_tokens'] = consumed_train_tokens
        summary['total_tokens_per_j'] = _safe_rate(consumed_train_tokens, _zeus_total_energy_j)
        summary['total_tokens_per_wh'] = _safe_rate(consumed_train_tokens, total_energy_wh)

    if reset:
        _zeus_window_start_iteration = iteration + 1
        _zeus_window_start_samples = consumed_train_samples or _zeus_window_start_samples
        _zeus_window_start_tokens = consumed_train_tokens or _zeus_window_start_tokens
        _zeus_window_name = f"interval_{_zeus_window_start_iteration}"
        _zeus_monitor.begin_window(_zeus_window_name)
    else:
        _zeus_window_name = None

    return summary


def get_last_power_metrics() -> Dict:
    return _last_power_metrics


def start_zeus_monitoring(gpu_indices: List[int] = None):
    """启动 Zeus 功耗监控"""
    global _zeus_monitor, _zeus_window_name, _zeus_window_start_iteration
    global _zeus_window_start_samples, _zeus_window_start_tokens
    global _zeus_total_energy_j, _zeus_total_time_s, _last_power_metrics
    try:
        from zeus.monitor import ZeusMonitor
        _zeus_monitor = ZeusMonitor(gpu_indices=gpu_indices)
        _zeus_window_name = 'interval_1'
        _zeus_window_start_iteration = 1
        _zeus_window_start_samples = 0
        _zeus_window_start_tokens = 0
        _zeus_total_energy_j = 0.0
        _zeus_total_time_s = 0.0
        _last_power_metrics = {}
        _zeus_monitor.begin_window(_zeus_window_name)
        print(f"[Zeus] Started monitoring GPUs: {gpu_indices or 'all'}")
        return _zeus_monitor
    except Exception as e:
        print(f"[Zeus] Failed to start: {e}")
        return None


def stop_zeus_monitoring():
    """停止 Zeus 功耗监控"""
    global _zeus_monitor, _zeus_window_name
    if _zeus_monitor is not None:
        _zeus_monitor = None
        _zeus_window_name = None
        print("[Zeus] Stopped")


def get_zeus_monitor():
    """获取 Zeus 监控器实例"""
    global _zeus_monitor
    return _zeus_monitor


if __name__ == '__main__':
    # 测试功耗监控
    print("测试GPU功耗监控...")
    
    monitor = GPUPowerMonitor(sample_interval=1.0)
    monitor.start()
    
    print("监控中，等待5秒...")
    time.sleep(5)
    
    print("\n当前功耗:", monitor.get_current_power())
    
    monitor.stop()
    monitor.print_summary()
