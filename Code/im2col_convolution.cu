#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
// im2col 核函数：将输入展开为列向量
__global__ void im2colKernel(const float* input, float* col, int padding_size, int kernel_size, 
                             int stride, int padding, int output_size, int depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 每个线程负责一个卷积核窗口
    if (tid < output_size * output_size) {
        int out_x = tid % output_size; // 输出矩阵的列索引
        int out_y = tid / output_size; // 输出矩阵的行索引
        for (int c = 0; c < depth; ++c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_x = out_x * stride + kx;
                    int in_y = out_y * stride + ky;
                    int col_index = ((c * kernel_size * kernel_size + ky * kernel_size + kx) * output_size * output_size) + tid;
                    int input_index = (c * padding_size * padding_size) + (in_y * padding_size + in_x);
                    col[col_index] = input[input_index];
                }
            }
        }
    }
}
// 矩阵乘法核函数（按通道累加）
__global__ void matrixMulKernel(const float* A, const float* B, float* C, 
                                int depth, int kernel_size, int output_size) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x < output_size && out_y < output_size) {
        int output_index = out_y * output_size + out_x;
        float sum = 0.0f;

        for (int c = 0; c < depth; ++c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int kernel_index = c * kernel_size * kernel_size + ky * kernel_size + kx;
                    int col_index = ((c * kernel_size * kernel_size + ky * kernel_size + kx) * output_size * output_size) + output_index;
                    sum += A[kernel_index] * B[col_index];
                }
            }
        }
        C[output_index] = sum;
    }
}
int main() {
    const int input_size = 8;    // 原始输入大小
    const int kernel_size = 3;   // 卷积核大小
    const int stride = 2;        // 步幅
    const int padding = 1;       // 填充大小
    const int depth = 3;         // 通道数量
    const int padding_size = input_size + 2 * padding; // 包含填充的输入大小
    const int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    auto start_time = std::chrono::high_resolution_clock::now();
    float* h_input = new float[padding_size * padding_size * depth];
    float* h_kernel = new float[kernel_size * kernel_size * depth];
    float* h_output = new float[output_size * output_size]();
    srand(time(0));
    for (int c = 0; c < depth; ++c) {
        for (int y = 0; y < padding_size; ++y) {
            for (int x = 0; x < padding_size; ++x) {
                int index = c * padding_size * padding_size + y * padding_size + x;

                if (x < padding || x >= input_size + padding || y < padding || y >= input_size + padding) {
                    h_input[index] = 0.0f; // 填充区域设置为 0
                } else {
                    h_input[index] = static_cast<float>(rand() % 10); // 非填充区域随机初始化
                }
            }
        }
    }
    for (int i = 0; i < kernel_size * kernel_size * depth; ++i) {
        h_kernel[i] = static_cast<float>(rand() % 10);
    }
    float *d_input, *d_kernel, *d_col, *d_output;
    const int col_size = output_size * output_size * kernel_size * kernel_size * depth;
    cudaMalloc(&d_input, padding_size * padding_size * depth * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * depth * sizeof(float));
    cudaMalloc(&d_col, col_size * sizeof(float));
    cudaMalloc(&d_output, output_size * output_size * sizeof(float));
    cudaMemcpy(d_input, h_input, padding_size * padding_size * depth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * depth * sizeof(float), cudaMemcpyHostToDevice);
    dim3 im2colBlockDim(256);
    dim3 im2colGridDim((output_size * output_size + im2colBlockDim.x - 1) / im2colBlockDim.x);
    // 执行 im2col 操作
    im2colKernel<<<im2colGridDim, im2colBlockDim>>>(d_input, d_col, padding_size, kernel_size, stride, padding, output_size, depth);
    cudaDeviceSynchronize();
    dim3 gemmBlockDim(16, 16);
    dim3 gemmGridDim((output_size + gemmBlockDim.x - 1) / gemmBlockDim.x,
                     (output_size + gemmBlockDim.y - 1) / gemmBlockDim.y);
    matrixMulKernel<<<gemmGridDim, gemmBlockDim>>>(d_kernel, d_col, d_output, depth, kernel_size, output_size);
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
    std::cout << "Convolution Result :\n";
    for (int y = 0; y < output_size; ++y) {
        for (int x = 0; x < output_size; ++x) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << h_output[y * output_size + x] << " ";
        }
        std::cout << std::endl;
    }
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_col);
    cudaFree(d_output);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Input_size:"<<input_size << std::endl;
    std::cout << "srtide:"<<stride << std::endl;
    std::cout << "Total Execution Time: " << duration.count() << " ms" << std::endl;
    return 0;
}