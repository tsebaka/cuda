%%cuda
#include <iostream>
#include <stdio.h>


__global__ void hist_kernel(int* input, int* hist, size_t n) {
    __shared__ unsigned int cache[256];
    cache[threadIdx.x] = 0;
    __syncthreads();
    
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        atomicAdd(&cache[input[i]], 1);
    }

    __syncthreads();
    atomicAdd(&(hist[threadIdx.x]), &(cache[threadIdx.x]));
}

int main() {
    size_t n = 10;
    int input[n] = {1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 7};
    int hist[n];
    int *dev_input;
    int *dev_hist;

    cudaMalloc((void**)&dev_input, n * sizeof(int));
    cudaMalloc((void**)&dev_hist, n * sizeof(int));

    cudaMemcpy(dev_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(1, 1);
    dim3 threadPerBlock(256);
    hist_kernel<<<blocksPerGrid, 256>>>(dev_input, dev_hist, n);

    cudaMemcpy(hist, dev_hist, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        std::cout << hist[i] << " "; 
    }

    cudaFree(dev_input);
    cudaFree(dev_hist);
    
    return 0;
}
