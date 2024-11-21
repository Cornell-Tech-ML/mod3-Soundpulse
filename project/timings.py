import random
from collections import defaultdict
import minitorch
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend, size=16) -> None:
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    sizes = list(times.keys())
    fast_times = [times[size]['fast'] for size in sizes]
    gpu_times = [times[size]['gpu'] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fast_times, 'o-', label='CPU Backend')
    plt.plot(sizes, gpu_times, 'o-', label='GPU Backend')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('MatMul Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # print()
    # print("Timing summary")
    # for size, stimes in times.items():
    #     print(f"Size: {size}")
    #     for b, t in stimes.items():
    #         print(f"    {b}: {t:.5f}")