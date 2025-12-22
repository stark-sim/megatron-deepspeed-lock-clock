#!/usr/bin/env python3
"""
测试不同GPU频率下NCCL all_reduce的时间
"""
import os
import sys
import time
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')

import torch
import torch.distributed as dist
import pynvml

def set_gpu_freq(freq_mhz):
    """设置所有GPU的频率"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq_mhz, freq_mhz)
        except Exception as e:
            print(f"GPU {i}: {e}")
    pynvml.nvmlShutdown()

def reset_gpu_freq():
    """重置GPU频率"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            pynvml.nvmlDeviceResetApplicationsClocks(handle)
        except:
            pass
    pynvml.nvmlShutdown()

def benchmark_all_reduce(tensor_size_mb, num_iters=20, warmup=5):
    """测量all_reduce时间"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 创建tensor (float32, 4 bytes per element)
    num_elements = int(tensor_size_mb * 1024 * 1024 / 4)
    tensor = torch.randn(num_elements, device=f'cuda:{local_rank}')
    
    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return sum(times) / len(times), min(times), max(times)

def main():
    # 初始化分布式
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Testing NCCL all_reduce at different GPU frequencies")
        print("=" * 70)
    
    # 测试不同大小的tensor
    tensor_sizes_mb = [500, 1000, 2000]  # MB
    
    # 测试不同频率
    frequencies = [1597, 1200, 1000, 800, 502]  # MHz
    
    for size_mb in tensor_sizes_mb:
        if rank == 0:
            print(f"\nTensor size: {size_mb} MB ({size_mb*1024*1024/4/1e6:.1f}M elements)")
            print("-" * 70)
        
        for freq in frequencies:
            dist.barrier()
            
            # 只有rank 0设置频率
            if rank == 0:
                set_gpu_freq(freq)
            dist.barrier()
            time.sleep(0.5)  # 等待频率稳定
            
            avg_time, min_time, max_time = benchmark_all_reduce(size_mb)
            
            if rank == 0:
                print(f"  {freq:4d} MHz: avg={avg_time:7.2f}ms, min={min_time:7.2f}ms, max={max_time:7.2f}ms")
            
            dist.barrier()
        
        # 重置频率
        if rank == 0:
            reset_gpu_freq()
        dist.barrier()
    
    if rank == 0:
        print("\n" + "=" * 70)
        reset_gpu_freq()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
