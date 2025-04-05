#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <iomanip>
#include <chrono>

// 检查 cuDNN 的返回状态
void checkCUDNN(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int input_size = 1024;    // 输入大小
    const int kernel_size = 3;   // 卷积核大小
    const int stride = 1;        // 步幅
    const int padding = 1;       // 填充大小
    const int depth = 3; // 输入通道数
    const int output_channels = 1; // 输出通道数
    const int padding_size = input_size + 2 * padding; // 包含填充的输入大小
    const int output_size = (input_size - kernel_size + 2 * padding) / stride + 1;
    const int batch_size = 1;    // 批量大小

    // 初始化 cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    // 创建 Tensor 描述符
    cudnnTensorDescriptor_t input_desc, output_desc;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_desc));
    // 输入 Tensor 描述
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, depth, input_size, input_size));
    // 输出 Tensor 描述
    checkCUDNN(cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
        batch_size, output_channels, output_size, output_size));
    // 卷积描述符
    cudnnConvolutionDescriptor_t conv_desc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc, padding, padding, stride, stride, 
        1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // 卷积核描述符
    cudnnFilterDescriptor_t filter_desc;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
        output_channels, depth, kernel_size, kernel_size));
   // 获取卷积算法
   cudnnConvolutionFwdAlgo_t algo;
   checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
       cudnn, input_desc, filter_desc, conv_desc, output_desc, 
       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
   // 获取工作空间大小
   size_t workspace_size = 0;
   checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
       cudnn, input_desc, filter_desc, conv_desc, output_desc, 
       algo, &workspace_size));
   // 分配工作空间
   void* workspace = nullptr;
   cudaMalloc(&workspace, workspace_size);
   // 分配输入、卷积核、输出内存
   float *d_input, *d_kernel, *d_output;
   cudaMalloc(&d_input, batch_size * depth * padding_size * padding_size * sizeof(float));
   cudaMalloc(&d_kernel, output_channels * depth * kernel_size * kernel_size * sizeof(float));
   cudaMalloc(&d_output, batch_size * output_channels * output_size * output_size * sizeof(float));
   // 初始化输入和卷积核（简化为随机数）
   float h_input[batch_size * depth * padding_size * padding_size];
   float h_kernel[output_channels * depth * kernel_size * kernel_size];
   // 初始化输入数据（包括填充区域）
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
   for (int i = 0; i < output_channels * depth * kernel_size * kernel_size; ++i) {
       h_kernel[i] = static_cast<float>(rand() % 10);
   }
   cudaMemcpy(d_input, h_input, batch_size * depth * padding_size * padding_size * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_kernel, h_kernel, output_channels * depth * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // 设置卷积前向操作的 alpha 和 beta
    float alpha = 1.0f, beta = 0.0f;

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行卷积前向操作
    checkCUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_desc, d_input, filter_desc, d_kernel, 
        conv_desc, algo, workspace, workspace_size, &beta, output_desc, d_output));

    // 停止计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);


    // 将结果拷贝回主机
    float h_output[batch_size * output_channels * output_size * output_size];
    cudaMemcpy(h_output, d_output, batch_size * output_channels * output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
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
    // 打印单通道输出结果
    // std::cout << "Single Channel Convolution Result:\n";
    // for (int y = 0; y < output_size; ++y) {
    //     for (int x = 0; x < output_size; ++x) {
    //         int idx = y * output_size + x;
    //         std::cout << std::setw(8) << std::fixed << std::setprecision(2) << h_output[idx] << " ";
    //     }
    //     std::cout << "\n";
    // }
    std::cout << "Input_size:"<<input_size << std::endl;
    std::cout << "srtide:"<<stride << std::endl;
    std::cout << "cuDNN Convolution Run Time: " << duration.count()  << " ms\n";

    // 清理资源
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
} 