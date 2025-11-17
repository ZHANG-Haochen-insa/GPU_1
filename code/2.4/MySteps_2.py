#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
练习2.4：增加算术复杂度
添加 MySillyFunction - 包含16个连续的数学运算
目标：增加计算密度，使 OpenCL 的优势得以发挥
"""

import numpy as np
import pyopencl as cl
import sys
import time

# ========================================
# 复杂数学函数（Numpy版本）
# ========================================
def MySillyFunction_Numpy(x):
    """
    应用16个连续的数学运算
    操作顺序：cos -> arccos -> sin -> arcsin -> tan -> arctan ->
             cosh -> arccosh -> sinh -> arcsinh -> tanh -> arctanh ->
             exp -> log -> sqrt -> square
    """
    # 为了数值稳定性，我们需要确保每个函数的输入在有效范围内
    x = np.abs(x) + 0.1  # 确保 x > 0，避免 log(0)
    x = np.cos(x)         # 结果在 [-1, 1]
    x = np.arccos(x)      # 结果在 [0, pi]
    x = np.sin(x)         # 结果在 [-1, 1]
    x = np.arcsin(x)      # 结果在 [-pi/2, pi/2]
    x = np.tan(x)         # 可能很大
    x = np.arctan(x)      # 结果在 (-pi/2, pi/2)
    x = np.abs(x) + 1.1   # 确保 > 1，为 arccosh 做准备
    x = np.cosh(x)        # 结果 >= 1
    x = np.arccosh(x)     # 结果 >= 0
    x = np.sinh(x)        # 可以是任意值
    x = np.arcsinh(x)     # 可以是任意值
    x = np.tanh(x)        # 结果在 (-1, 1)
    x = x * 0.9           # 缩放到 (-0.9, 0.9) 为 arctanh 做准备
    x = np.arctanh(x)     # 结果是任意值
    x = np.abs(x) + 0.1   # 确保 > 0
    x = np.exp(x)         # 结果 > 0
    x = np.log(x)         # 结果可以是任意值
    x = np.abs(x)         # 确保 >= 0
    x = np.sqrt(x)        # 结果 >= 0
    x = x * x             # 平方
    return x

# ========================================
# OpenCL 内核代码
# ========================================
OPENCL_KERNEL = """
// MySillyFunction 的 OpenCL 实现
float MySillyFunction(float x) {
    x = fabs(x) + 0.1f;
    x = cos(x);
    x = acos(x);
    x = sin(x);
    x = asin(x);
    x = tan(x);
    x = atan(x);
    x = fabs(x) + 1.1f;
    x = cosh(x);
    x = acosh(x);
    x = sinh(x);
    x = asinh(x);
    x = tanh(x);
    x = x * 0.9f;
    x = atanh(x);
    x = fabs(x) + 0.1f;
    x = exp(x);
    x = log(x);
    x = fabs(x);
    x = sqrt(x);
    x = x * x;
    return x;
}

__kernel void sum(
    __global const float *a_g,
    __global const float *b_g,
    __global float *res_g)
{
    int gid = get_global_id(0);
    float a = MySillyFunction(a_g[gid]);
    float b = MySillyFunction(b_g[gid]);
    res_g[gid] = a + b;
}
"""

# ========================================
# 本地计算函数（使用Numpy）
# ========================================
def NativeAddition(a, b):
    """
    使用Numpy进行复杂向量加法
    先对 a 和 b 应用 MySillyFunction，然后相加
    """
    start_time = time.time()
    a_transformed = MySillyFunction_Numpy(a)
    b_transformed = MySillyFunction_Numpy(b)
    res = a_transformed + b_transformed
    end_time = time.time()
    time_elapsed = end_time - start_time
    return res, time_elapsed

# ========================================
# OpenCL计算函数
# ========================================
def OpenCLAddition(a_np, b_np):
    """
    使用OpenCL进行复杂向量加法
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

    # 编译OpenCL内核程序
    prg = cl.Program(ctx, OPENCL_KERNEL).build()

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
        vector_size = 1024  # 默认大小

    print(f"========================================")
    print(f"练习 2.4：增加算术复杂度")
    print(f"向量大小: {vector_size}")
    print(f"每个元素执行 16 个数学运算")
    print(f"========================================\n")

    # 生成随机输入数据
    a_np = np.random.rand(vector_size).astype(np.float32)
    b_np = np.random.rand(vector_size).astype(np.float32)

    # 调用本地计算函数
    print("执行本地计算（Numpy + MySillyFunction）...")
    res_np, native_time = NativeAddition(a_np, b_np)
    native_rate = vector_size / native_time if native_time > 0 else 0
    # 计算总操作数：每个元素 16 次运算 × 2 个向量 + 1 次加法 = 33 次运算
    native_ops = vector_size * 33
    native_flops = native_ops / native_time if native_time > 0 else 0
    print(f"  执行时间: {native_time:.6f} 秒")
    print(f"  性能速率: {native_rate:.0f} 元素/秒")
    print(f"  性能速率: {native_rate/1e6:.2f} M元素/秒")
    print(f"  计算性能: {native_flops/1e6:.2f} MFlops\n")

    # 调用OpenCL计算函数
    print("执行OpenCL计算（MySillyFunction）...")
    res_cl, opencl_time = OpenCLAddition(a_np, b_np)
    opencl_rate = vector_size / opencl_time if opencl_time > 0 else 0
    opencl_ops = vector_size * 33
    opencl_flops = opencl_ops / opencl_time if opencl_time > 0 else 0
    print(f"  执行时间: {opencl_time:.6f} 秒")
    print(f"  性能速率: {opencl_rate:.0f} 元素/秒")
    print(f"  性能速率: {opencl_rate/1e6:.2f} M元素/秒")
    print(f"  计算性能: {opencl_flops/1e6:.2f} MFlops\n")

    # 计算性能比率
    performance_ratio = opencl_rate / native_rate if native_rate > 0 else 0
    speedup = native_time / opencl_time if opencl_time > 0 else 0

    print(f"========================================")
    print(f"性能分析")
    print(f"========================================")
    print(f"OpenCL/Native 性能比率: {performance_ratio:.6f}")
    if performance_ratio > 1:
        print(f"  ✓ OpenCL 比 Native 快 {performance_ratio:.2f} 倍")
    else:
        print(f"  ✗ Native 比 OpenCL 快 {1/performance_ratio:.2f} 倍")
    print(f"加速比: {speedup:.6f}x")
    print(f"OpenCL FlOPS 提升: {opencl_flops/native_flops:.2f}x")
    print(f"========================================\n")

    # 验证结果正确性（由于浮点运算顺序可能不同，允许一定误差）
    diff = res_cl - res_np
    norm = np.linalg.norm(diff)
    max_diff = np.max(np.abs(diff))
    rel_error = norm / np.linalg.norm(res_np) if np.linalg.norm(res_np) > 0 else 0

    print("结果验证:")
    print(f"  差值范数: {norm:.6e}")
    print(f"  最大差值: {max_diff:.6e}")
    print(f"  相对误差: {rel_error:.6e}")

    # 由于复杂的浮点运算，误差会稍大
    if rel_error < 1e-3:  # 0.1% 相对误差
        print("  ✓ 计算成功！OpenCL结果与Numpy结果在可接受误差范围内。")
    else:
        print(f"  ⚠ 警告：相对误差较大，可能存在数值问题。")
