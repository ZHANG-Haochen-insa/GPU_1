# Exercise 3.1 Test Report

## 1. Objective

This test aims to evaluate the performance and behavior of the initial PyCUDA implementation developed in Exercise 3.1. The main objectives include:
1.  Comparing the performance of the PyCUDA implementation with Numpy (native) and PyOpenCL.
2.  Verifying and reproducing the inherent limitations of the "incorrect" `threadIdx.x`-based implementation as described in the tutorial.

## 2. Methodology

We executed the `code/3.1/MySteps_3.py` script with a range of different vector sizes (from 32 to 32768) as input. For each run, we recorded the performance rates (in "Million elements/second" or M-elem/s) for all three implementations (Numpy, OpenCL, CUDA).

## 3. Test Results

The table below summarizes the performance data collected from the test logs:

| Vector Size | Numpy Rate (M-elem/s) | OpenCL Rate (M-elem/s) | CUDA Rate (M-elem/s) |
| :---------- | :-------------------- | :--------------------- | :------------------- |
| 32          | 2.63                  | 0.00                   | 0.00                 |
| 64          | 7.90                  | 0.00                   | 0.01                 |
| 128         | 11.93                 | 0.00                   | 0.01                 |
| 256         | 29.02                 | 0.00                   | 0.02                 |
| 512         | 42.95                 | 0.00                   | 0.05                 |
| 1024        | 104.76                | 0.00                   | 0.08                 |
| 2048        | 182.76                | 0.01                   | **Failed**           |
| 4096        | 464.32                | 0.02                   | **Failed**           |
| 8192        | 818.09                | 0.03                   | **Failed**           |
| 16384       | 1676.08               | 0.07                   | **Failed**           |
| 32768       | 2694.88               | 0.14                   | **Failed**           |

## 4. Analysis

### 4.1. Performance Observations

For all successful executions (vector sizes ≤ 1024), **Numpy significantly outperformed both OpenCL and CUDA**. This is entirely expected due to:
-   **Low computational density**: Simple vector addition involves minimal computation.
-   **Overhead**: For GPU implementations (OpenCL and CUDA), the overhead of transferring data from host memory (CPU) to device memory (GPU), launching the kernel, and then transferring results back to the host, far outweighs any potential gains from parallel computation on the GPU.

Consequently, in such scenarios, using a GPU not only fails to provide a performance boost but actually leads to a significant performance degradation.

### 4.2. CUDA Implementation Failure

The test results clearly show that the CUDA implementation **consistently fails when the vector size is strictly greater than 1024**. The error logged was:
`cuFuncSetBlockShape failed: invalid argument`

The reason for this error, as explained in the tutorial, is:
-   The code uses `block=(int(a_np.size), 1, 1)` to define the thread block dimensions for kernel launch.
-   This instructs CUDA to launch a **single thread block** (implied by `grid=(1,1)` in the `sum_kernel` call) containing `a_np.size` number of threads.
-   However, there is a hardware-imposed limit on the number of threads a single CUDA thread block can contain, which is typically **1024**.
-   When the requested number of threads (e.g., 2048) exceeds this limit, the CUDA driver rejects the request, deeming it an "invalid argument," and thus throwing the `cuFuncSetBlockShape failed: invalid argument` error.

## 5. Conclusion

This test successfully validated the objectives of Exercise 3.1:
1.  It demonstrated that for operations with low arithmetic complexity, the overhead of GPU execution makes it less efficient than native CPU implementations.
2.  It successfully reproduced the execution failure caused by the incorrect use of CUDA's parallel hierarchy (attempting to place more threads than allowed within a single block).

These results underscore the critical importance of understanding and correctly utilizing CUDA's two-tiered parallel hierarchy: **Blocks** and **Threads**. The next exercise (3.2) will address this fundamental limitation by employing multiple blocks to distribute the workload.