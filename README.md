# Cuda-parallel-program

一个基于CUDA并行编程的项目，包含通用矩阵乘法（GEMM）的并行实现、直接卷积与im2col优化卷积的GPU实现，以及CUBLAS和cuDNN库的性能对比分析。通过实验探索不同并行计算策略的性能差异，并总结优化方法。

## 项目文件
- **Code文件夹**：存储项目所有代码
  - `matrix_multiply.cu`：CUDA实现的通用矩阵乘法（支持动态调整线程块大小）
  - `convolution.cu`：基于CUDA的直接卷积实现（支持多通道、步幅与填充）
  - `im2col_convolution.cu`：im2col结合GEMM的卷积优化实现
  - `cublas.cu`：CUBLAS库的矩阵乘法性能对比
  - `cuDNN.cu`：cuDNN库的卷积实现与性能测试
- **Result文件夹**：存储实验结果
  - 子任务1为cuda并行化通用矩阵乘法
  - 子任务2为与cublas调库矩阵乘法进行性能对比分析
  - 子任务3为cuda并行化直接卷积法
  - 子任务4为利用cuda结合im2col和GEE来并行化卷积
  - 子任务5为cuDNN调库卷积进行性能对比分析
- **Report.pdf**：完整实验报告（含代码分析、性能图表与优化总结）

## 实验内容

### 子任务1：CUDA通用矩阵乘法（GEMM）
- **目标**：实现可扩展的并行矩阵乘法，支持矩阵规模（512-8192）和线程块大小（32~512）
- **核心方法**：
  - 使用二维线程网格划分输出矩阵
  - 全局内存直接访问优化（`matrixMulKernel`函数）
  - 动态资源分配与异步数据传输
- **关键指标**：不同矩阵规模下线程块大小对运行时间的影响

### 子任务2：CUBLAS库性能对比
- **目标**：对比自定义CUDA GEMM与CUBLAS库的性能差异
- **实现**：
  - 使用`cublasSgemm`接口实现矩阵乘法
  - 统一内存管理策略（主机端与设备端内存分配）
- **发现**：CUBLAS在8192矩阵规模下性能提升达15-20倍

### 子任务3：直接卷积实现
- **目标**：实现多通道2D卷积（支持stride=1/2/3）
- **关键技术**：
  - 三维线程索引设计（x/y轴定位输出，z轴遍历输入通道）
  - 动态填充计算（`padding_size = input_size + 2*padding`）
  - 分支优化：通过条件判断避免越界访问
- **性能瓶颈**：全局内存频繁访问导致延迟

### 子任务4：im2col优化卷积
- **目标**：通过矩阵乘法加速卷积计算
- **流程**：
  1. `im2colKernel`将输入展开为列矩阵
  2. 修改GEMM核函数实现通道累加
- **优势**：复用高效矩阵乘法，减少条件判断
- **局限**：显存占用随输入尺寸平方增长

### 子任务5：cuDNN卷积实现
- **目标**：集成cuDNN库并对比性能
- **关键步骤**：
  - 使用`cudnnConvolutionForward`接口
  - 自动算法选择（`CUDNN_CONVOLUTION_FWD_PREFER_FASTEST`）
  - Tensor描述符与卷积参数配置
- **性能表现**：在512x512输入下比直接卷积快27倍

---

## 快速开始

### 环境配置
```bash
# 基础依赖
sudo apt install build-essential
# CUDA Toolkit 11.7
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run
# cuDNN 8.5（需官网下载）
tar -xzvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/* /usr/local/cuda/include/
sudo cp cudnn-*-archive/lib/* /usr/local/cuda/lib64/
