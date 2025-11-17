#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
练习3.2：使用Block进行CUDA并行化
目标：修改CUDA实现，通过使用多个Block（每个Block一个Thread）来处理任意大小的向量，
      解决练习3.1中大于1024个元素就失败的问题。
"""

import numpy as np
import sys
import time

# 尝试导入OpenCL和CUDA库
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# ========================================
# OpenCL 内核代码
# ========================================
OPENCL_KERNEL = """
__kernel void sum(
    __global const float *a_g,
    __global const float *b_g,
    __global float *res_g)
{
    int gid = get_global_id(0);
    res_g[gid] = a_g[gid] + b_g[gid];
}
"""

# ========================================
# CUDA 内核代码 (练习3.2修改版)
# ========================================
CUDA_KERNEL = """
__global__ void sum(float *a, float *b, float *c)
{
  // 使用 blockIdx.x 作为索引
  int idx = blockIdx.x;
  c[idx] = a[idx] + b[idx];
}
"""

# ========================================
# 本地计算函数（使用Numpy）
# ========================================
def NativeAddition(a, b):
    start_time = time.time()
    res = a + b
    end_time = time.time()
    time_elapsed = end_time - start_time
    return res, time_elapsed

# ========================================
# OpenCL计算函数
# ========================================
def OpenCLAddition(a_np, b_np):
    if not OPENCL_AVAILABLE:
        return None, 0

    start_time = time.time()
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    prg = cl.Program(ctx, OPENCL_KERNEL).build()
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    knl = prg.sum
    knl(queue, a_np.shape, None, a_g, b_g, res_g)
    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)
    queue.finish()
    end_time = time.time()
    time_elapsed = end_time - start_time
    return res_np, time_elapsed

# ========================================
# CUDA计算函数 (练习3.2修改版)
# ========================================
def CudaAddition(a_np, b_np):
    if not CUDA_AVAILABLE:
        return None, 0

    start_time = time.time()
    
    mod = SourceModule(CUDA_KERNEL)
    sum_kernel = mod.get_function("sum")

    a_gpu = cuda.mem_alloc(a_np.nbytes)
    b_gpu = cuda.mem_alloc(b_np.nbytes)
    res_gpu = cuda.mem_alloc(a_np.nbytes)

    cuda.memcpy_htod(a_gpu, a_np)
    cuda.memcpy_htod(b_gpu, b_np)

    # 执行内核（练习3.2修改版）
    # block=(1,1,1): 每个Block只有1个Thread
    # grid=(size,1): 启动'size'个Block
    sum_kernel(a_gpu, b_gpu, res_gpu, block=(1, 1, 1), grid=(int(a_np.size), 1))

    res_np = np.empty_like(a_np)
    cuda.memcpy_dtoh(res_np, res_gpu)
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    return res_np, time_elapsed

# ========================================
# 主程序
# ========================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        vector_size = int(sys.argv[1])
    else:
        # 默认大小，用于演示
        vector_size = 32768

    print(f"========================================")
    print(f"练习 3.2: 使用Block进行CUDA并行化")
    print(f"向量大小: {vector_size}")
    print(f"========================================\n")

    a_np = np.random.rand(vector_size).astype(np.float32)
    b_np = np.random.rand(vector_size).astype(np.float32)

    # --- 本地计算 ---
    res_native, native_time = NativeAddition(a_np, b_np)
    native_rate = vector_size / native_time if native_time > 0 else 0
    print(f"Numpy 计算:")
    print(f"  - 执行时间: {native_time:.6f} 秒")
    print(f"  - 性能速率: {native_rate/1e6:.2f} M元素/秒\n")

    # --- OpenCL 计算 ---
    if OPENCL_AVAILABLE:
        res_cl, opencl_time = OpenCLAddition(a_np, b_np)
        opencl_rate = vector_size / opencl_time if opencl_time > 0 else 0
        print(f"OpenCL 计算:")
        print(f"  - 执行时间: {opencl_time:.6f} 秒")
        print(f"  - 性能速率: {opencl_rate/1e6:.2f} M元素/秒")
        if np.allclose(res_native, res_cl):
            print("  - ✓ 结果验证成功")
        else:
            print("  - ✗ 结果验证失败")
        print("")
    else:
        print("OpenCL 未安装，跳过计算。\n")

    # --- CUDA 计算 ---
    if CUDA_AVAILABLE:
        try:
            res_cuda, cuda_time = CudaAddition(a_np, b_np)
            cuda_rate = vector_size / cuda_time if cuda_time > 0 else 0
            print(f"CUDA 计算:")
            print(f"  -执行时间: {cuda_time:.6f} 秒")
            print(f"  - 性能速率: {cuda_rate/1e6:.2f} M元素/秒")
            if np.allclose(res_native, res_cuda):
                print("  - ✓ 结果验证成功")
            else:
                print("  - ✗ 结果验证失败")
        except Exception as e:
            print(f"CUDA 计算失败:")
            print(f"  - 错误: {e}")
        print("")
    else:
        print("PyCUDA 未安装，跳过计算。\n")

    print("========================================")
    print("执行完成。")
    print("========================================")
