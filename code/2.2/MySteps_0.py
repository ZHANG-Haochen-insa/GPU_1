#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
练习2.2：重组代码
将代码分为本地计算和OpenCL计算两个函数
"""

import numpy as np
import pyopencl as cl

# ========================================
# 本地计算函数（使用Numpy）
# ========================================
def NativeAddition(a, b):
    """
    使用Numpy进行向量加法
    输入: a, b - numpy数组
    输出: res - 相加结果
    """
    res = a + b
    return res

# ========================================
# OpenCL计算函数
# ========================================
def OpenCLAddition(a_np, b_np):
    """
    使用OpenCL进行向量加法
    输入: a_np, b_np - numpy数组
    输出: res_np - OpenCL计算结果
    """
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

    return res_np

# ========================================
# 主程序
# ========================================
if __name__ == "__main__":
    # 生成随机输入数据
    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    # 调用本地计算函数
    res_np = NativeAddition(a_np, b_np)

    # 调用OpenCL计算函数
    res_cl = OpenCLAddition(a_np, b_np)

    # 比较两种方法的结果
    print(res_cl - res_np)
    print(np.linalg.norm(res_cl - res_np))
    assert np.allclose(res_cl, res_np)

    print("计算成功！OpenCL结果与Numpy结果一致。")
