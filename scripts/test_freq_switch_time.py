#!/usr/bin/env python3
"""
测试频率切换开销：串行 vs 并行
"""
import sys
import time
sys.path.insert(0, '/home/sd/.local/lib/python3.10/site-packages')

import pynvml
from concurrent.futures import ThreadPoolExecutor

def test_serial_switch(handles, low_freq, high_freq, num_iters=20):
    """串行切换频率"""
    times = []
    for _ in range(num_iters):
        # 切换到低频
        start = time.perf_counter()
        for idx, handle in handles:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, low_freq, low_freq)
        low_time = (time.perf_counter() - start) * 1000
        
        # 切换到高频
        start = time.perf_counter()
        for idx, handle in handles:
            pynvml.nvmlDeviceSetGpuLockedClocks(handle, high_freq, high_freq)
        high_time = (time.perf_counter() - start) * 1000
        
        times.append((low_time, high_time))
    
    avg_low = sum(t[0] for t in times) / len(times)
    avg_high = sum(t[1] for t in times) / len(times)
    return avg_low, avg_high

def test_parallel_switch(handles, low_freq, high_freq, num_iters=20):
    """并行切换频率"""
    executor = ThreadPoolExecutor(max_workers=len(handles))
    
    def set_freq(args):
        idx, handle, freq = args
        pynvml.nvmlDeviceSetGpuLockedClocks(handle, freq, freq)
        return idx
    
    times = []
    for _ in range(num_iters):
        # 切换到低频
        start = time.perf_counter()
        tasks = [(idx, handle, low_freq) for idx, handle in handles]
        list(executor.map(set_freq, tasks))
        low_time = (time.perf_counter() - start) * 1000
        
        # 切换到高频
        start = time.perf_counter()
        tasks = [(idx, handle, high_freq) for idx, handle in handles]
        list(executor.map(set_freq, tasks))
        high_time = (time.perf_counter() - start) * 1000
        
        times.append((low_time, high_time))
    
    executor.shutdown(wait=False)
    avg_low = sum(t[0] for t in times) / len(times)
    avg_high = sum(t[1] for t in times) / len(times)
    return avg_low, avg_high

def main():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    handles = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        handles.append((i, handle))
    
    # 获取支持的频率
    _, first_handle = handles[0]
    mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(first_handle)
    mem_clock = mem_clocks[0]
    supported_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(first_handle, mem_clock)
    
    high_freq = supported_clocks[0]  # 1597 MHz
    low_freq = min(supported_clocks, key=lambda x: abs(x - 1000))  # ~1000 MHz
    
    print(f"Testing frequency switching on {device_count} GPUs")
    print(f"High freq: {high_freq} MHz, Low freq: {low_freq} MHz")
    print("=" * 60)
    
    # 测试串行
    print("\nSerial switching (20 iterations)...")
    serial_low, serial_high = test_serial_switch(handles, low_freq, high_freq)
    print(f"  Low->High: {serial_high:.2f}ms, High->Low: {serial_low:.2f}ms")
    print(f"  Total round-trip: {serial_low + serial_high:.2f}ms")
    
    # 测试并行
    print("\nParallel switching (20 iterations)...")
    parallel_low, parallel_high = test_parallel_switch(handles, low_freq, high_freq)
    print(f"  Low->High: {parallel_high:.2f}ms, High->Low: {parallel_low:.2f}ms")
    print(f"  Total round-trip: {parallel_low + parallel_high:.2f}ms")
    
    print("\n" + "=" * 60)
    speedup = (serial_low + serial_high) / (parallel_low + parallel_high)
    print(f"Speedup: {speedup:.2f}x")
    
    # 重置频率
    for idx, handle in handles:
        pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        pynvml.nvmlDeviceResetApplicationsClocks(handle)
    
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
