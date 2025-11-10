# INSA 2025: GPU实践教程

## 课程概述

本实践课程由4小时的课时组成。作为INSA Lyon的最后一次课程，在2025年10月17日和11月10日由Emmanuel Quémener教授讲授了两次预备课程：
- CPU和并行架构课程
- GPU：21世纪的颠覆性技术

## CQQCOQP：如何？谁？何时？多少？哪里？什么？为什么？

- **为什么？** 全面了解GPU并掌握研究方法
- **什么？** 在简单示例上编程、测试和比较GPU
- **何时？** 从2025年11月10日开始
- **多少？** 测量GPU相比其他机器提供的性能
- **哪里？** 在工作站、集群节点、笔记本电脑（配置良好的）、终端上
- **谁？** 适合学生、教师、研究人员、好奇的技术人员
- **如何？** 通过应用一些简单命令，通常在终端中执行

## 课程目标

掌握机器中的GPU，理解OpenCL和CUDA编程，通过几个简单示例和生产代码比较CPU与GPU的性能。

## 实践课程安排

1. ENS-Lyon Blaise Pascal中心远程环境的使用
2. 硬件发现，包括CPU和GPU
3. 使用Python/OpenCL文档的基础示例逐步探索
4. Python/CUDA的中场插曲以测试GPU上的另一种实现
5. 离散傅里叶变换的实现和移植
6. Python中的设备选择及其编程
7. 使用外部库：xGEMM示例
8. 集成"业务代码"：TensorFlow、GENESIS和Gromacs
9. 使用矩阵代码进行性能测量

为了保留您的工作记录并能够评估，要求编写一份"日志"，基于提出的问题。请截图并将其集成到您的文档中，以及您产生的代码。

---

## 第1章：GPU硬件调查

### 1.1 我的机器里有什么？

在科学计算中，硬件由冯·诺伊曼架构定义：
- **CPU**（中央处理单元）：带有CU（控制单元）和ALU（算术逻辑单元）
- **MU**（内存单元）
- **输入输出设备**

GPU通常被视为输入/输出设备。像大多数安装在机器中的设备一样，它们使用PCI或PCI Express互连总线。

#### 练习1.1：获取(GP)GPU设备列表

使用命令 `lspci -nn | egrep '(VGA|3D)'` 获取GPU设备信息。

```bash
lspci -nn | egrep '(VGA|3D)'
```

**任务：**
1. 列出了多少VGA设备？
2. 列出了多少3D设备？
3. 获取GPU电路的扩展型号名称
4. 从网上检索每个GPU的以下信息：
   - 计算单元数量（"cuda cores"或"stream processors"）
   - 计算核心的基础频率
   - 内存频率

示例输出：
```
3b:00.0 VGA compatible controller [0300]: NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] [10de:1b06] (rev a1)
a1:00.0 VGA compatible controller [0300]: NVIDIA Corporation GK107GL [Quadro K420] [10de:0ff3] (rev a1)
```

#### 练习1.2：使用dmesg获取机器信息

```bash
dmesg | grep -i nvidia
```

**任务：**
1. 内核加载了哪个驱动程序版本？
2. 如果存在，`input: HDA NVidia`设备代表什么？
3. 它是图形设备吗？

#### 练习1.3：使用lsmod获取主机信息

```bash
lsmod | grep nvidia
```

**任务：**
1. 信息是否与上述相同？逐字符比较？

#### 练习1.4：使用ls获取机器信息

```bash
ls -l /dev/nvidia*
```

**任务：**
1. 您有多少个`/dev/nvidia<number>`？
2. 此信息是否与前3个练习一致？

#### 练习1.5：使用nvidia-smi获取信息

```bash
nvidia-smi
```

示例输出：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 0000:82:00.0      On |                  N/A |
| 23%   31C    P8    10W / 250W |     35MiB / 11172MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

**任务：**
1. 识别上述特征并比较元素
2. 列出了多少进程？

#### 练习1.6：使用clinfo获取信息

```bash
clinfo -l
```

OpenCL设备按平台/设备层次结构呈现。

示例输出：
```
Platform #0: AMD Accelerated Parallel Processing
 `-- Device #0: Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz
Platform #1: Portable Computing Language
 `-- Device #0: pthread-Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz
Platform #2: NVIDIA CUDA
 +-- Device #0: GeForce GTX 1080 Ti
 `-- Device #1: Quadro K420
Platform #3: Intel(R) OpenCL
 `-- Device #0: Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz
```

**OpenCL实现详解：**
- **#0,#0** AMD加速并行处理：AMD的CPU实现，最古老，性能接近OpenMP
- **#1,#0** Portable Computing Language：开源CPU实现，效率不高
- **#2,#0** Nvidia CUDA：Nvidia的CUDA实现，设备0，GeForce GTX 1080 Ti
- **#2,#1** Nvidia CUDA：Nvidia的CUDA实现，设备1，Quadro K420
- **#3,#0** Intel(R) OpenCL：Intel的CPU实现，相当高效

**任务：**
1. 识别并比较您的输出与上述列表
2. 您有多少图形设备？

#### 练习1.7：使用clinfo获取详细信息

```bash
clinfo | egrep '(Platform Name|Device Name|Max compute|Max clock)'
```

**任务：**
1. 比较CPU实现之间的信息。为什么有这些差异？
2. 将CPU的处理单元数与网络数据比较：Intel的Ark
3. 将GPU的处理单元数与网络数据比较：Nvidia规格站点或Wikipedia
4. 比较识别的频率与网络上找到的频率
5. 您是否发现Compute Units数量与cuda cores数量之间的一致性？
6. 每个Compute Unit包含多少cuda cores？

**关键概念：**
- 对于CPU：处理单元通常是物理核心数乘以线程数的乘积
- 对于GPU：这是宏处理单元的数量：Nvidia的SM（流多处理器）单元，AMD/ATI的CU（计算单元）单元

#### 练习1.8：使用CUDA_VISIBLE_DEVICES选择GPU

可以使用环境变量`CUDA_VISIBLE_DEVICES`选择使用哪个Nvidia GPU。

```bash
# 仅使用nvidia-smi识别的GPU #0
CUDA_VISIBLE_DEVICES=1 <MyProgram>

# 仅使用nvidia-smi识别的GPU #1
CUDA_VISIBLE_DEVICES=0 <MyProgram>

# 使用GPU #0和#1
CUDA_VISIBLE_DEVICES=0,1 <MyProgram>

# 不使用任何GPU
CUDA_VISIBLE_DEVICES='' <MyProgram>
```

**任务：**
1. 尝试 `CUDA_VISIBLE_DEVICES=0 clinfo -l` 并观察输出
2. 尝试 `CUDA_VISIBLE_DEVICES=1 clinfo -l` 并观察输出
3. 尝试 `CUDA_VISIBLE_DEVICES=0,1 clinfo -l` 并观察输出
4. 尝试 `CUDA_VISIBLE_DEVICES='' clinfo -l` 并观察输出
5. 您是否观察到不同设备的选择？

#### 练习1.9：监控系统资源

在实践中，重点关注CPU或GPU在执行期间的资源占用。

```bash
# 在一个终端中
dstat -cim

# 在另一个终端中
nvidia-smi dmon
```

**任务：**
1. 打开终端，输入 `dstat -cim` 并观察输出
2. 详细说明参数c、i和m的用途
3. 打开终端，输入 `nvidia-smi dmon` 并观察输出
4. 详细说明选项dmon的用途
5. 使用Ctrl+C停止执行
6. 使用 `-i 0` 或 `-i 1` 重新启动命令
7. 详细说明选项 `-i` 后跟整数的用途

### 1.2 获取源代码

几乎所有CBP用于比较CPU和GPU的工具都在bench4xpu项目中。

```bash
git clone https://github.com/numa65536/bench4xpu
```

**目录结构：**
- **BLAS**：包含xGEMM和xTRSV：测试所有BLAS库实现
- **Epidevomath**：GPU上项目实现的原型（已放弃）
- **FFT**：cuFFT的首次使用（暂停）
- **Ising**：Python中Ising模型的多种实现（多重并行化）
- **NBody**：牛顿N体模型的OpenCL实现
- **Pi**：Pi蒙特卡洛的多种实现
- **Splutter**：内存喷射器模型，用于评估原子函数
- **TrouNoir**：1994年代码的移植示例，1997年移植到C，2019年移植到Python/OpenCL和Python/CUDA
- **ETSN**：ETSN 2022暑期学校的校正程序

---

## 第2章：Python和OpenCL的首次探索

基于PyOpenCL官方文档的程序。它将两个向量`a_np`和`b_np`相加为向量`res_np`。

```python
#!/usr/bin/env python

import numpy as np
import pyopencl as cl

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
knl = prg.sum
knl(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# 在CPU上用Numpy检查：
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))
assert np.allclose(res_np, a_np + b_np)
```

### 练习2.1：首次执行

**任务：**
1. 使用编辑器（例如gedit）
2. 复制/粘贴上述程序源代码
3. 以名称MySteps.py保存源代码
4. 启动它并判断执行：`python MySteps.py`
5. 启动它并判断执行：`python3 MySteps.py`
6. 更改MySteps.py的执行权限
7. 直接用`./MySteps.py`启动它
8. 如果启动失败，修改MySteps.py
9. 使用PYOPENCL_CTX的所有组合为前缀启动
   - 例如：`PYOPENCL_CTX=X:Y ./MySteps.py`
10. 将标准输出重定向到文件MySteps_XY.out
11. (X,Y)定义为(platform,device)
12. 注意，如果只有X而没有Y，只指定X
13. 例如：`PYOPENCL_CTX=X:Y ./MySteps.py > MySteps_XY.out 2>&1`

### 2.1 重组代码

这个演示示例将被深度修改，转变为矩阵代码。我们将：
1. 通过隔离部分（特别是与OpenCL相关的部分）对代码进行注释
2. 将本地计算过程提取到函数NativeAddition中
3. 将完整的OpenCL过程提取到函数OpenCLAddition中
4. 调用NativeAddition函数以找到结果res_np
5. 调用OpenCLAddition函数以找到结果res_cl
6. 使用结果res_np和res_cl修改测试

### 练习2.2：在不更改输出的情况下修改

**任务：**
1. 根据上述6项规范修改MySteps_0.py
2. 为多个设备执行程序
3. 为每次执行保存标准输出
4. 使用diff命令比较练习2.1和2.2的输出

### 2.2 性能测量

MySteps_1.py将集成以下修改：
1. 可以将向量大小作为执行参数传递
2. 本地命令的执行时间
3. OpenCL命令的执行时间
4. 不同大小的本地执行速度估计
5. 不同大小的OpenCL执行速度估计
6. 本地模式和OpenCL模式之间的性能比率
7. 释放OpenCLAddition块中预留的内存

### 练习2.3：代码的最小仪表化

**任务：**
1. 根据上述7项规范修改MySteps_1.py
2. 对2^15到2^30的向量大小执行程序
   - 在最强大的GPU上
   - 在最有效的CPU实现上：Intel
3. 分析在哪些情况下出现问题
4. 将这些困难与硬件规格联系起来
5. 用这些结果填写表格
6. 总结OpenCL在这种使用情况下的效率

**重要观察：**
- 简单的加法操作中，OpenCL实现永远不会比原生Python更快
- 原生Python执行始终快4-6倍
- 但是，随着向量大小增加，OpenCL性能比率不断提高

**关键结论：**
使用OpenCL（在CPU和GPU上）需要：
- 执行基本操作的元素数量相当大（从几千到几百万，取决于计算设备）
- 每个元素执行的基本操作数量具有"足够"的算术密度（大于十）

### 2.3 增加计算密度

从MySteps_1.py开始，复制到MySteps_2.py。我们将集成一个连续堆叠16个操作的函数：cos、arccos、sin、arcsin、tan、arctan、cosh、arccosh、sinh、arcsinh、tanh、arctanh、exp、log、sqrt，最后是平方。

这个函数命名为MySillyFunction，必须集成到原生Python和OpenCL内核中。在加两个向量时，我们将在加之前对a和b的元素应用此函数。

### 练习2.4：增加算术复杂度

**任务：**
1. 根据8项规范修改MySteps_2.py
2. 对大小为32（即2^5）执行程序
   - 在最强大的GPU上
   - 在最有效的CPU实现上：Intel
3. 保存前两次执行的输出
4. 重新执行前两次执行
5. 保存两次执行的输出
6. 您对OpenCL合成的持续时间有何发现？
7. 对从32到33554432的向量大小执行程序
8. 分析在哪些情况下出现问题
9. 用这些结果填写表格

**性能结果示例（GTX Titan）：**

对于GPU（6GB RAM的GTX Titan）：
| Size | NativeRate | OpenCLRate | Ratio |
|------|------------|------------|-------|
| 1024 | 1087884 | 3351 | 0.003080 |
| 1048576 | 1535449 | 3529675 | 2.298790 |
| 33554432 | 1484349 | 52485826 | 35.359492 |

对于CPU（Intel实现）：
| Size | NativeRate | OpenCLRate | Ratio |
|------|------------|------------|-------|
| 1024 | 1082128 | 3099 | 0.002864 |
| 1048576 | 1318155 | 3223043 | 2.445117 |
| 33554432 | 1474004 | 22517796 | 15.276618 |

**结论：**
当向量大小接近百万时，转向OpenCL的增益是显著的。CPU加速15倍，GPU加速35倍。如果显著增加负载，CPU增益达到21倍，GPU超过127倍！

---

## 第3章：回到C及其OpenMP和OpenACC实现

### 3.1 C语言实现

在并行编程领域（在CPU或GPU上），OpenCL的使用是边缘的，从Python调用它更是如此...我们将通过示例展示这是一个应该冷静批评的错误。

为此，让我们以我们的简单加法程序MySteps_1.py为例。它在ETSN目录中有其C实现MySteps_1.c。

**编译：**
```bash
gcc -O3 -o MySteps_1 MySteps_1.c -lm
```

### 3.2 OpenMP实现

OpenMP编程的基本原则是"打破循环"，或者更准确地说，将数组元素上的独立计算分配给可用资源。为此，OpenMP依赖于源代码的"标记"：`#pragma`。

**编译：**
```bash
gcc -fopenmp -O3 -o MySteps_1_openmp MySteps_1_openmp.c -lm -lgomp
```

无串行执行的编译：
```bash
gcc -DNOSERIAL -fopenmp -O3 -o MySteps_1_openmp_NoSerial MySteps_1_openmp.c -lm -lgomp
```

### 3.3 OpenACC实现

OpenACC编程的原理与OpenMP非常接近：利用标记来识别要发送到外部设备（GPU或加速器）的源代码中的计算。

**编译：**
```bash
gcc -O3 -fopenacc -foffload=nvptx-none -foffload="-O3 -misa=sm_35 -lm" -o MySteps_1_openacc MySteps_1_openacc.c -lm
```

无串行执行的编译：
```bash
gcc -DNOSERIAL -O3 -fopenacc -foffload=nvptx-none -foffload="-O3 -misa=sm_35 -lm" -o MySteps_1_openacc_NoSerial MySteps_1_openacc.c -lm
```

### 性能比较

在gtxtitan机器上（GTX Titan和E6-2620 CPU，6核2 GHz）的性能比较：

| Size | C/Serial | C/OpenMP | C/OpenACC | Numpy | PyCL CPU | PyCL GPU |
|------|----------|----------|-----------|-------|----------|----------|
| 1024 | 1024000000 | 2151260 | 2039 | 69273666 | 3060 | 2952 |
| 33554432 | 928919488 | 1800710016 | 157982016 | 640650623 | 60821580 | 57072436 |

**结论：**
- C/OpenMP在向量大小超过100万时表现最佳
- 对于小尺寸（小于32768），串行C仍然是最有效的
- Python/Numpy在32768到1048576之间的大小上仍然非常有竞争力
- OpenACC比OpenCL/CPU和OpenCL/GPU实现高出3倍

---

## 第4章：CUDA插曲及其PyCUDA实现

### 4.1 CUDA简介

Nvidia很早就感受到为其GPU提供简单编程抽象的必要性。2002年就推出了cg-toolkit。2007年夏天才有了完整的语言。

今天，CUDA在制造商的库中无处不在，但也存在于绝大多数其他开发中。然而，它的问题来自于对制造商的依赖：CUDA只服务于Nvidia。

Andreas Kloeckner除了开发PyOpenCL外，还开发了PyCUDA，通过Python以类似的方式利用CUDA。

### 练习3.1：PyCUDA实现

从MySteps_2.py复制到MySteps_3.py。

**任务：**
1. 根据3项规范修改MySteps_3.py
2. 对从32到32768的向量大小执行程序
3. 分析在哪些情况下出现问题
4. 将这些困难与硬件规格联系起来
5. 用这些结果填写表格
6. 总结CUDA在这种使用情况下的效率

**问题：**
CUDA实现仅适用于大小≤1024的向量...这个限制实际上是由于CUDA的错误使用。

CUDA（在较小程度上OpenCL）有2个并行化级别：
- **Threads**（线程）：最细粒度的并行化级别
- **Blocks**（块）：更高级别的并行化

在CUDA和OpenCL实现中，最大可调用线程数仅为1024！

### 练习3.2：使用Blocks

修改MySteps_4.py：
1. 在CUDA内核中将threadIdx替换为blockIdx
2. 在sum调用中：将`block=(a_np.size,1,1)`替换为`block=(1,1,1)`
3. 在sum调用中：将`grid=(1,1)`替换为`grid=(a_np.size)`

### 练习3.3：增加计算复杂度

从MySteps_4.py复制到MySteps_5.py，并添加MySillyFunction的CUDA实现。

**性能结果：**
| Size | NativeRate | OpenCL Rate | CUDA Rate | OpenCL ratio | CUDA ratio |
|------|------------|-------------|-----------|--------------|------------|
| 32768 | 1220822 | 104351 | 29276 | 0.085476 | 0.023981 |
| 268435456 | 1485632 | 102563944 | 12149328 | 69.037247 | 8.177885 |

**结论：**
CUDA的增益是实质性的，但仍远低于OpenCL。

---

## 第5章：实现"昂贵"函数：离散傅里叶变换

### 5.1 Python"天真"实现

离散傅里叶变换在信号处理中无处不在。它特别允许比直接卷积更有效的滤波操作。

从MySteps_5.py复制到MyDFT_1.py。

### 练习5.1：Python"天真"实现

**任务：**
1. 根据7项规范修改MyDFT_1.py
2. 对大小为16执行程序并检查一致性
3. 对从16到4096的向量大小执行程序
4. 在表中放置性能

### 5.2 Python Numpy实现

双重迭代在计算方面特别昂贵。第二个实现将利用Numpy的广播函数来避免这种双循环，将其限制为单循环。

### 练习5.2：Python Numpy实现

复制MyDFT_1.py为MyDFT_2.py并修改。

### 5.3 Numba实现

Numba的使用提供了类似于OpenMP的并行化可能性。

### 练习5.3：Python Numba实现

### 5.4 OpenCL实现

从MyDFT_3.py复制到MyDFT_4.py。

**性能结果（1024元素）：**
- CPU（Intel实现）：
  - Numpy: 6282
  - Numba: 356
  - OpenCL: 3306

### 5.5 CUDA实现

从MyDFT_4.py复制到MyDFT_5.py。

**性能对比：**
| Size | NumpyRate | NumbaRate | OpenCL Rate | CUDA Rate |
|------|-----------|-----------|-------------|-----------|
| 1024 | 6387 | 422 | 3239 | 7480 |
| 65536 | 221 | 1214 | 11162 | 13446 |

---

## 第6章：OpenCL和CUDA中的设备选择

### 6.1 选择方法

在第1章中，我们看到可以使用环境变量CUDA_VISIBLE_DEVICES选择CUDA设备。在第2章中，我们看到为了避免指定OpenCL设备，我们可以使用环境变量PYOPENCL_CTX。

现在我们将看到系统化的方法，可以在任何PyOpenCL或PyCUDA程序中使用。

### 练习6.1：探索PiXPU.py

**任务：**
1. 识别对应于默认参数的行
2. 识别OpenCL设备发现的行
3. 识别CUDA设备发现的行
4. 识别提取输入参数的行
5. 识别执行时OpenCL设备选择的行
6. 识别执行时CUDA设备选择的行

### 练习6.2-6.4：修改MyDFT程序

从MyDFT_5.py开始，逐步添加设备选择功能。

---

## 第7章：使用GPU的"核心"探索：xGEMM

### 7.1 从BLAS到xGEMM

在科学计算中，目标是不要在每次数值建模时重新发明轮子。近40年来，最常见的线性代数库是BLAS（基本线性代数子程序）。

BLAS库中矩阵乘法的实现是xGEMM，其中x替换为S、D、C和Z，分别用于单精度（32位）、双精度（64位）、复数和单精度以及复数和双精度。

### 练习7.1：编辑xGEMM.c源代码

### 练习7.2：启动各种xGEMM实现

**性能对比示例：**
- 单精度GPU（GTX 1080 Ti）：7597 GFlops
- 单精度CPU（OpenBLAS）：366 GFlops
- GPU比最佳CPU快约20倍

### 练习7.3：不同大小的性能

---

## 第8章：探索"业务代码"集成

### 8.1 TensorFlow深度学习

使用TensorFlow利用CIFAR10图像数据库进行卷积学习的教程。

### 练习8.1：执行CIFAR10教程

### 练习8.2：修改网络架构

### 8.2 GENESIS代码集成与利用

GENESIS是RIKEN计算科学中心的分子动力学软件。它是一个混合程序（利用2种并行化策略），甚至是三混合程序：
- 使用CUDA利用Nvidia类型的GPU
- 通过OpenMP在核心上分布
- 通过MPI消息传递在不同节点上分布

### 练习9.1：检索并编译代码

### 练习9.2：执行alad_water示例

### 8.3 Gromacs代码集成与利用

分子动力学软件Gromacs。

### 练习10.1：应用Nvidia的"配方"

### 练习10.2：执行示例1536

---

## 第9章：使用Pi蒙特卡洛探索GPU

### 9.1 "Pi蒙特卡洛"或"Pi飞镖冲刺"

通过蒙特卡洛方法计算Pi在几个方面都是示范性的：
- 简单：随机数生成器、测试机制和计数器就足够了
- 可并行化：通过在计算单元上分配迭代次数
- 计算密集型：几乎没有内存访问（保留在计算寄存器中）
- 但是，它对于计算Pi是出了名的低效 ;-)

### 练习10.1：探索OpenCL代码

### 练习10.2-10.6：各种并行度和精度的性能测试

**关键发现：**
- GPU仅在高并行度下显示其功率
- GTX 1080 Ti比最佳CPU实现快33倍
- 双精度性能下降显著

### 9.2 Python/OpenCL和Python/CUDA实现

使用PiXPU.py程序。

### 练习11.1-11.4：代码检查和可扩展性研究

---

## 第10章：细粒度N体探索

NBody.py程序，位于bench4xpu/NBody中，原理是在牛顿模型中确定每个粒子与所有其他粒子相互作用的位置和速度。

这是一个细粒度代码：每次迭代时，每个粒子的每个位置和每个速度都将被所有其他粒子的接近度修改。

### 练习12.1：研究NBody.py的源代码

### 练习12.2：32768粒子启动NBody.py

**性能结果示例：**
- GTX 1080 Ti（单精度）：104 GSquertz
- 最佳CPU（单精度）：4.7 GSquertz
- GPU比CPU快约22倍

### 练习12.3：以-g模式启动NBody.py

可以使用-g选项执行NBody.py以获得实时计算动画。

---

## 结论

正如您在这些实践练习中注意到的那样，利用可能充满惊喜：相关的性能测量不能脱离所利用硬件的知识。

"业务代码"的利用也让您瞥见了在环境中集成和执行程序的难度，即使在非常同质的环境中：所有利用的工作站都有完全相同的操作系统SIDUS。仅允许执行程序的"技巧"也说明，没有经验，很难解决。
