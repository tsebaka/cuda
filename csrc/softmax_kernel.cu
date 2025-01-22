%%cuda
#include <iostream>
#include <stdio.h>
#include <cmath>

const int blocksPerGrid = 16;
const int threadsPerBlock = 16;


extern "C" __global__ 
void softmax_kernel(float* input, float* output, size_t n) {
    __shared__ float cache[threadsPerBlock];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float temp = 0;
    for (int i = tid; i < n; i += stride) {
        temp += std::exp(input[i]);
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
            //printf("cache index: %d, value: %d\n", cacheIndex, cache[cacheIndex]);
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        output[blockIdx.x] = cache[i];
    }
}

int main() {
    size_t n = 50;
    float input[n];
    float output[blocksPerGrid];
    float* dev_input;
    float* dev_output;

    for (int i = 0; i < n; ++i) {
        input[i] = (float)std::log(i + 1);
        std::cout << i + 1 << " " << input[i] << "\n"; 
    }
    
    cudaMalloc((void**)&dev_input, n * sizeof(float));
    cudaMalloc((void**)&dev_output, blocksPerGrid * sizeof(float));

    cudaMemcpy(dev_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_input, dev_output, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(output, dev_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocksPerGrid; ++i)
        std::cout << output[i] << " ";

    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0;
}
