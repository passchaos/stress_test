import numpy as np
import time
import cupy as cp
from cupyx.profiler import benchmark

def max_memory_bandwidth_test(dtype=cp.float32, iterations=100):
    free_mem, total_mem = cp.cuda.Device().mem_info
    available_mem = free_mem  - 1 * 1024**3
    elem_size = cp.dtype(dtype).itemsize

    max_n = int((available_mem / (2 * elem_size))**0.5)
    max_n = (max_n // 512) * 512
    print(f"使用矩阵尺寸：{max_n}x{max_n} {dtype}")
    print(f"单矩阵内存：{max_n*max_n*elem_size / 1024**3:.2f} GB")

    a = cp.random.rand(max_n, max_n, dtype=dtype)
    # time.sleep(10)
    print("after create a")
    b = cp.random.rand(max_n, max_n, dtype=dtype)
    print("after create b")
    # time.sleep(5)

    c = cp.random.rand(max_n, 10, dtype=dtype)
    print("begin warmup matmul")

    cp.matmul(a, c)
    a += b
    cp.cuda.Stream.null.synchronize()
    print("begin compute")

    start_time = time.time()
    for _ in range(iterations):
        a += b
        b *= 0.5
        cp.get_default_memory_pool().free_all_blocks()

    cp.cuda.Stream.null.synchronize()
    end_time = time.time()

    total_elements = 5 * max_n * max_n * iterations
    total_bytes = total_elements * elem_size
    elapsed_time = end_time - start_time

    bandwidth = (total_bytes / 1024**3) / elapsed_time
    print(f"总数据量：{total_bytes / 1024**3:.2f} GB")
    print(f"耗时：{elapsed_time:.2f} 秒")
    print(f"实际内存带宽：{bandwidth:.2f} GB/s")

    return bandwidth

def my_func(arr):
    new_arr = [cp.sqrt(cp.sum(a**2, axis=-1)) for a in arr]
    # return cp.sqrt(cp.sum(a**2, axis=-1))
    return new_arr

def gpu_loop():
    while True:
        start = time.perf_counter()

        print(f"begin gemm logic")
        for i in range(10):
            m, n, k = 4096, 5120, 4096
            a = cp.random.rand(m, n)
            b = cp.random.rand(n, k)

            print(f"begin matmul: {i}")
            res = cp.matmul(a, b)

            del a, b, res

        elapsed = time.perf_counter() - start

        print(f"GPU is still running, the load loop took in {elapsed} seconds")

        cp.get_default_memory_pool().free_all_blocks()


    # count = 0

    # a = b = np.array(np.random.rand(vec_size, vec_size, vec_size, vec_size), dtype=np.float64)
    # c = np.zeros(vec_size, dtype=np.float64)

    # while True:
    #     start = timer()
    #     c = pow_gpu(a, b)
    #     print("GPU is still running, the load loop took in {0} seconds".format(timer() - start))


def cpu_loop():
    vec_size = 10000000

    while True:
        a = np.zeros_like((vec_size, vec_size))

if __name__ == '__main__':
    max_memory_bandwidth_test(dtype=cp.float32, iterations=100)
    # try:
    #     arr = []
    #     for _ in range(10):
    #         a = cp.random.random((4096, 5120))
    #         arr.append(a)

    #     while True:
    #         print(benchmark(my_func, (arr,), n_repeat=20))
    #         time.sleep(10)
    #     # gpu_loop()
    # except KeyboardInterrupt:
    #     print("\nCtrl+C detected. Exiting gracefully.")
    # finally:
    #     print("Cleanup complete.")
    # mp.set_start_method('spawn')

    # gpu = Process(target=gpu_loop)
    # gpu.start()

    # cpu = Process(target=cpu_loop)
    # cpu.start()

    # gpu.join()
    # cpu.join()
