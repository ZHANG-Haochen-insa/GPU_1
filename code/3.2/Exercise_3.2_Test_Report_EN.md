# Exercise 3.2 Test Report

## 1. Objective

This test aims to evaluate the corrected PyCUDA implementation from Exercise 3.2, which circumvents the thread-per-block limit by assigning an independent thread block to each element. The main objectives are:
1.  To verify that the new CUDA implementation can successfully process large-scale vectors with more than 1024 elements.
2.  To re-evaluate the performance of Numpy, OpenCL, and the corrected CUDA implementation on a larger data scale.

## 2. Methodology

We executed the `MySteps_4.py` script located in the `code/3.2/` directory, using a series of large-scale vectors ranging from 32,768 to 16,777,216 elements as input. For each run, we recorded the performance rate (in "Million elements/second" or M-elem/s) for all three implementations.

## 3. Test Results

The table below summarizes the performance data collected from the test logs:

| Vector Size | Numpy Rate (M-elem/s) | OpenCL Rate (M-elem/s) | CUDA Rate (M-elem/s) |
| :---------- | :-------------------- | :--------------------- | :------------------- |
| 32,768      | 2545.17               | 0.14                   | 0.11                 |
| 65,536      | 4294.97               | 0.27                   | 5.52                 |
| 131,072     | 4196.61               | 0.57                   | 9.52                 |
| 262,144     | 6704.34               | 1.09                   | 12.55                |
| 524,288     | 6745.47               | 2.12                   | 36.29                |
| 1,048,576   | 6346.39               | 4.32                   | 75.73                |
| 2,097,152   | 4627.09               | 8.82                   | 136.00               |
| 4,194,304   | 2739.79               | 17.50                  | 176.79               |
| 8,388,608   | 2441.49               | 33.28                  | 274.55               |
| 16,777,216  | 2352.92               | 62.61                  | 311.00               |

## 4. Analysis

### 4.1. Implementation Success

Most importantly, **the corrected CUDA implementation successfully processed all tested vector sizes**, including those far exceeding 1024. This proves that by distributing the parallelization task across multiple blocks (the Grid dimension) instead of within a single block (the Block dimension), we have successfully bypassed the hardware limitation on the number of threads per block.

### 4.2. Performance Observations

-   **Numpy is Still Fastest**: Despite the significant increase in data scale, for a simple, memory-bandwidth-bound operation like vector addition, Numpy (backed by efficient C libraries) remains the top performer.
-   **CUDA vs. OpenCL**: After the correction, the CUDA implementation's performance far exceeds that of OpenCL. As the vector size increases, CUDA's performance advantage becomes more pronounced. At the largest tested scale (~16M elements), the CUDA rate is approximately 5 times faster than the OpenCL rate (311.00 vs 62.61 M-elem/s). This suggests that in this "one-thread-per-element" model, PyCUDA's implementation or the CUDA driver's overhead management is more efficient than PyOpenCL's.
-   **Scalability**: The performance (M-elem/s) of both GPU implementations steadily improves as the data volume increases. This shows that the GPU architecture exhibits good scalability for this type of massively parallel task by amortizing fixed costs (like kernel launch and memory copy) over more computational units.

## 5. Conclusion

The test for Exercise 3.2 was a success. We have verified that by leveraging thread blocks for parallelization, we can build a scalable CUDA program capable of handling arbitrary data sizes.

However, while the scalability of the GPU implementations is validated, their absolute performance for a simple vector addition is still far from matching the CPU-based Numpy. This again highlights the importance of **arithmetic intensity**. To truly leverage the massive computational power of the GPU, we need each thread to perform more computational work, not just a single addition.

This paves the way for Exercise 3.3, which will focus on increasing the arithmetic intensity of the kernel to observe the resulting performance changes.