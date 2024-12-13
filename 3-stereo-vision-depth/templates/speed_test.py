import numpy as np
import time

# Create a 1000x1000 grayscale image (array)
image = np.random.rand(10000, 10000)

# Row-major iteration
start = time.time()
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel = image[i, j]
row_major_time = time.time() - start

# Column-major iteration
start = time.time()
for j in range(image.shape[1]):
    for i in range(image.shape[0]):
        pixel = image[i, j]
col_major_time = time.time() - start

speedup = col_major_time / row_major_time
print(f"Row-major time: {row_major_time:.5f} sec")
print(f"Column-major time: {col_major_time:.5f} sec")
print(f"Speedup: {speedup:.2f}x")
