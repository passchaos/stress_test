import numpy as np
import time

if __name__ == "__main__":
    max_n = 4096 * 1
    # max_n = 81920

    rng = np.random.default_rng()

    dtype = np.float32

    a = rng.random((max_n, max_n), dtype=dtype)
    b = rng.random((max_n, max_n), dtype=dtype)

    elem_size = np.dtype(dtype).itemsize

    while True:
        begin = time.time()
        a += b
        b *= 0.5

        elapsed_time = time.time() - begin
        total_bytes = 5 * max_n * max_n * elem_size

        bandwidth = (total_bytes / 1024**3) / elapsed_time
        print(f"占用内存带宽：{bandwidth:.2f} GB/s")
        # time.sleep(0.1)
