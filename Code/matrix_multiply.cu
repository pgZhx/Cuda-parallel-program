#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

__global__ void matrixMulKernel(float *A, float *B, float *C, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % N; // 输出矩阵C的列索引
    int y = idx / N; // 输出矩阵C的行索引

    if (x < N && y < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[y * N + i] * B[i * K + x];
        }
        C[y * N + x] = sum;
    }
}

int main() {
    int M, N, K;
    M = N = K = 512;
    int blocksize = 32;
    float *h_A = new float[M * N];
    float *h_B = new float[N * K];
    float *h_C = new float[M * K];
    srand(time(0));
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = static_cast<float>(rand()%10);
    }
    for (int i = 0; i < N * K; ++i) {
        h_B[i] = static_cast<float>(rand()%10) ;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * K * sizeof(float));
    cudaMalloc((void **)&d_C, M * K * sizeof(float));
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(blocksize);
    dim3 numBlocks((M * K + threadsPerBlock.x - 1) / threadsPerBlock.x);    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, K);
    cudaDeviceSynchronize();
    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //验证矩阵乘法正确性
    // std::cout << "Matrix A:"<< std::endl;
    // for (int y = 0; y <M; ++y) {
    //     for (int x = 0; x < N; ++x) {
    //         // 设置每个元素占8个字符宽度，保留1位小数
    //         std::cout << std::setw(8) << std::fixed << std::setprecision(1) << h_A[y * M + x] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Matrix B:"<< std::endl;
    // for (int y = 0; y <M; ++y) {
    //     for (int x = 0; x < N; ++x) {
    //         // 设置每个元素占8个字符宽度，保留1位小数
    //         std::cout << std::setw(8) << std::fixed << std::setprecision(1) << h_B[y * M + x] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "Matrix C:"<< std::endl;
    // for (int y = 0; y <M; ++y) {
    //     for (int x = 0; x < N; ++x) {
    //         // 设置每个元素占8个字符宽度，保留1位小数
    //         std::cout << std::setw(8) << std::fixed << std::setprecision(1) << h_C[y * M + x] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    std::cout << "Run time: " << elapsed.count() << " s" << std::endl;
    std::cout << "Matrix sizes: "<< M << std::endl;
    std::cout << "Block sizes:" << blocksize << std::endl;

    return 0;
}