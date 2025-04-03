%%cuda
#include <iostream>
#include <stdio.h>


__global__ void hist_kernel(
    int* input, 
    int* hist,
    const size_t n,
    const int bins,
    const int max,
    const int min,
    const float h
) {
    extern __shared__ unsigned int cache[];

    if (threadIdx.x < bins) {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        int bin = int((input[i] - min) / h);
        if (bin >= bins) {
            bin = bins - 1;
        }
        atomicAdd(&cache[bin], 1);
    }

    __syncthreads();
    if (threadIdx.x < bins) {
        atomicAdd(&(hist[threadIdx.x]), cache[threadIdx.x]);
    }
}

int find_max(const int* input, const size_t n) {
    int max_value = -10000;
    for (int i = 0; i < n; ++i) {
        if (input[i] > max_value)
            max_value = input[i];
    }
    return max_value;
}

int find_min(const int* input, const size_t n) {
    int min_value = 10000;
    for (int i = 0; i < n; ++i) {
        if (input[i] < min_value)
            min_value = input[i];
    }
    return min_value;
}

int main() {
    const size_t n = 10;
    const int bins = 3;
    // int input[n] = {1, 1, 1, 2, 2, 3, 3, 4, 5, 5};
    int* input = (int*)std::malloc(n * sizeof(int));
    input[0] = 1;
    input[1] = 1;
    input[2] = 1;
    input[3] = 2;
    input[4] = 2;
    input[5] = 3;
    input[6] = 3;
    input[7] = 4;
    input[8] = 5;
    input[9] = 5;

    int hist[bins];
    int *dev_input;
    int *dev_hist;

    const int maxx = find_max(input, n);
    const int minn = find_min(input, n);
    const float h = float((maxx - minn + 1)) / bins;

    cudaMalloc((void**)&dev_input, n * sizeof(int));
    cudaMalloc((void**)&dev_hist, bins * sizeof(int));

    cudaMemcpy(dev_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(56);
    dim3 threadPerBlock(256);
    size_t sharedMemSize = bins * sizeof(unsigned int);
    hist_kernel<<<blocksPerGrid, threadPerBlock, sharedMemSize>>>(dev_input, dev_hist, n, bins, maxx, minn, h);

    cudaMemcpy(hist, dev_hist, bins * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < bins; ++i) {
        std::cout << hist[i] << " "; 
    }

    delete[] input;
    cudaFree(dev_input);
    cudaFree(dev_hist);
    
    return 0;
}
