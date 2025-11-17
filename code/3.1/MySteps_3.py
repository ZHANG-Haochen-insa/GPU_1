#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
练习3.1：集成PyCUDA实现
目标：在现有框架中添加PyCUDA实现，并与Numpy、PyOpenCL进行性能比较。
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
# CUDA 内核代码
# ========================================
CUDA_KERNEL = """
__global__ void sum(float *a, float *b, float *c)
{
  int tid = threadIdx.x;
  c[tid] = a[tid] + b[tid];
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
# CUDA计算函数
# ========================================
def CudaAddition(a_np, b_np):
    if not CUDA_AVAILABLE:
        return None, 0

    start_time = time.time()
    
    # 编译CUDA内核
    mod = SourceModule(CUDA_KERNEL)
    sum_kernel = mod.get_function("sum")

    # 分配设备内存
    a_gpu = cuda.mem_alloc(a_np.nbytes)
    b_gpu = cuda.mem_alloc(b_np.nbytes)
    res_gpu = cuda.mem_alloc(a_np.nbytes)

    # 将数据从主机复制到设备
    cuda.memcpy_htod(a_gpu, a_np)
    cuda.memcpy_htod(b_gpu, b_np)

    # 执行内核（练习3.1的特定方式，效率较低）
    # 警告：当 a_np.size > 1024 时，这将失败
    sum_kernel(a_gpu, b_gpu, res_gpu, block=(int(a_np.size), 1, 1), grid=(1, 1))

    # 将结果从设备复制回主机
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
        vector_size = 1024

    print(f"========================================")
    print(f"练习 3.1: PyCUDA 初步集成")
    print(f"向量大小: {vector_size}")
    print(f"========================================\n")

    # 生成随机数据
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
            print(f"  - 执行时间: {cuda_time:.6f} 秒")
            print(f"  - 性能速率: {cuda_rate/1e6:.2f} M元素/秒")
            # 围绕allclose的异常处理
            if np.allclose(res_native, res_cuda):
                print("  - ✓ 结果验证成功")
            else:
                print("  - ✗ 结果验证失败")
        except Exception as e:
            print(f"CUDA 计算失败:")
            print(f"  - 错误: {e}")
            print("  - 这在向量大小 > 1024 时是预期的行为。")
        print("")
    else:
        print("PyCUDA 未安装，跳过计算。\n")

    print("========================================")
    print("执行完成。")
    print("========================================")