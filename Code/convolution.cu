#include <cuda_runtime.h>
#include <iostream>
#include <iomanip> // 用于格式化输出
#include <cstdlib>
#include <ctime>
#include <chrono>

// CUDA卷积核函数
__global__ void convolutionKernel(const float* input, const float* kernel, float* output, 
                                  int padding_size, int kernel_size, int output_size, int stride, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < output_size && y < output_size) {
        float value = 0.0f;
        // 遍历通道
        for (int c = 0; c < depth; ++c) {
            // 遍历卷积核
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    // 计算输入图像中的位置
                    int ix = x * stride + kx;
                    int iy = y * stride + ky;
                    if (ix >= 0 && ix < padding_size && iy >= 0 && iy < padding_size) {
                        int input_idx = (c * padding_size * padding_size) + (iy * padding_size + ix);
                        int kernel_idx = (c * kernel_size * kernel_size) + (ky * kernel_size + kx);
                        value += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        output[y * output_size + x] = value;
    }
}

int main() {
    const int input_size = 8; 
    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int depth = 3;
    const int padding_size = input_size + 2 * padding; // 包含填充后的大小
    const int output_size = (input_size - kernel_size + 2 * padding) / stride + 1; // 输出矩阵的尺寸
    auto start_time = std::chrono::high_resolution_clock::now();
    float* h_input = new float[padding_size * padding_size * depth];
    float* h_kernel = new float[kernel_size * kernel_size * depth];
    float* h_output = new float[output_size * output_size]();
    // 初始化输入数据（带填充）
    srand(time(0));
    for (int c = 0; c < depth; ++c) {
        for (int y = 0; y < padding_size; ++y) {
            for (int x = 0; x < padding_size; ++x) {
                int index = c * padding_size * padding_size + y * padding_size + x;
                if (x < padding || x >= input_size + padding || y < padding || y >= input_size + padding) {
                    h_input[index] = 0.0f; // 填充区域
                } else {
                    h_input[index] = static_cast<float>(rand() % 10); // 非填充区域
                }
            }
        }
    }
    // 初始化卷积核
    for (int i = 0; i < kernel_size * kernel_size * depth; ++i) {
        h_kernel[i] = static_cast<float>(rand() % 5 + 1); // 随机权重
    }
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, padding_size * padding_size * depth * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * depth * sizeof(float));
    cudaMalloc(&d_output, output_size * output_size * sizeof(float));
    cudaMemcpy(d_input, h_input, padding_size * padding_size * depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * depth * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16);
    dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x,
                 (output_size + blockDim.y - 1) / blockDim.y);

    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                             padding_size, kernel_size, output_size, stride, depth);

    cudaMemcpy(h_output, d_output, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "Inpu(include padding)t:\n";
    // for(int c = 0;c < depth;c++){
    //     std::cout <<"Channel"<< c<< ":"<< std::endl;
    //     for (int y = 0; y < padding_size; ++y) {
    //         for (int x = 0; x < padding_size; ++x) {
    //             // 设置每个元素占8个字符宽度，保留1位小数
    //             std::cout << std::setw(8) << std::fixed << std::setprecision(1) << h_input[padding_size*padding_size*c+y * padding_size + x] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << "Kernel:\n";
    // for(int c = 0;c < depth;c++){
    //     std::cout <<"Channel"<< c<< ":"<< std::endl;
    //     for (int y = 0; y <kernel_size; ++y) {
    //         for (int x = 0; x < kernel_size; ++x) {
    //             // 设置每个元素占8个字符宽度，保留1位小数
    //             std::cout << std::setw(8) << std::fixed << std::setprecision(1) << h_kernel[c*kernel_size * kernel_size +  y * kernel_size + x] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    std::cout << "Convolution Result:\n";
    for (int y = 0; y < output_size; ++y) {
        for (int x = 0; x < output_size; ++x) {
            // 设置每个元素占8个字符宽度，保留1位小数
            std::cout << std::setw(8) << std::fixed << std::setprecision(1) << h_output[y * output_size + x] << " ";
        }
        std::cout << std::endl;
    }
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Input_size:"<<input_size << std::endl;
    std::cout << "srtide:"<<stride << std::endl;
    std::cout << "Total Execution Time: " << duration.count() << " ms" << std::endl;
    return 0;
}