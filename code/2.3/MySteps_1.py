#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
练习2.3：代码的最小仪表化
添加性能测量功能：
1. 命令行参数（向量大小）
2. 执行时间测量
3. 性能速率计算
4. 性能比率分析
5. 内存释放
"""

import numpy as np
import pyopencl as cl
import sys
import time

# ========================================
# 本地计算函数（使用Numpy）
# ========================================
def NativeAddition(a, b):
    """
    使用Numpy进行向量加法
    输入: a, b - numpy数组
    输出: res - 相加结果, time_elapsed - 执行时间（秒）
    """
    start_time = time.time()
    res = a + b
    end_time = time.time()
    time_elapsed = end_time - start_time
    return res, time_elapsed

# ========================================
# OpenCL计算函数
# ========================================
def OpenCLAddition(a_np, b_np):
    """
    使用OpenCL进行向量加法
    输入: a_np, b_np - numpy数组
    输出: res_np - OpenCL计算结果, time_elapsed - 执行时间（秒）
    """
    start_time = time.time()

    # 创建OpenCL上下文
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # 创建内存标志
    mf = cl.mem_flags

    # 将主机数据复制到设备内存
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    # 定义并编译OpenCL内核程序
    prg = cl.Program(ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()

    # 创建输出缓冲区
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

    # 执行内核
    knl = prg.sum
    knl(queue, a_np.shape, None, a_g, b_g, res_g)

    # 将结果从设备复制回主机
    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # 确保所有操作完成
    queue.finish()

    end_time = time.time()
    time_elapsed = end_time - start_time

    # 释放内存
    a_g.release()
    b_g.release()
    res_g.release()

    return res_np, time_elapsed

# ========================================
# 主程序
# ========================================
if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        vector_size = int(sys.argv[1])
    else:
        vector_size = 50000  # 默认大小

    print(f"========================================")
    print(f"向量大小: {vector_size}")
    print(f"========================================\n")

    # 生成随机输入数据
    a_np = np.random.rand(vector_size).astype(np.float32)
    b_np = np.random.rand(vector_size).astype(np.float32)

    # 调用本地计算函数
    print("执行本地计算（Numpy）...")
    res_np, native_time = NativeAddition(a_np, b_np)
    native_rate = vector_size / native_time if native_time > 0 else 0
    print(f"  执行时间: {native_time:.6f} 秒")
    print(f"  性能速率: {native_rate:.0f} 元素/秒")
    print(f"  性能速率: {native_rate/1e6:.2f} M元素/秒\n")

    # 调用OpenCL计算函数
    print("执行OpenCL计算...")
    res_cl, opencl_time = OpenCLAddition(a_np, b_np)
    opencl_rate = vector_size / opencl_time if opencl_time > 0 else 0
    print(f"  执行时间: {opencl_time:.6f} 秒")
    print(f"  性能速率: {opencl_rate:.0f} 元素/秒")
    print(f"  性能速率: {opencl_rate/1e6:.2f} M元素/秒\n")

    # 计算性能比率
    performance_ratio = opencl_rate / native_rate if native_rate > 0 else 0
    speedup = opencl_time / native_time if opencl_time > 0 else 0

    print(f"========================================")
    print(f"性能分析")
    print(f"========================================")
    print(f"OpenCL/Native 性能比率: {performance_ratio:.6f}")
    if performance_ratio > 1:
        print(f"  -> OpenCL 比 Native 快 {performance_ratio:.2f} 倍")
    else:
        print(f"  -> Native 比 OpenCL 快 {1/performance_ratio:.2f} 倍")
    print(f"时间比率: {speedup:.6f}")
    print(f"========================================\n")

    # 验证结果正确性
    diff = res_cl - res_np
    norm = np.linalg.norm(diff)
    print("结果验证:")
    print(f"  差值范数: {norm}")

    if norm < 1e-5:
        print("  ✓ 计算成功！OpenCL结果与Numpy结果一致。")
    else:
        print(f"  ✗ 警告：结果存在差异！")
        print(f"  最大差值: {np.max(np.abs(diff))}")
        print(f"  平均差值: {np.mean(np.abs(diff))}")
