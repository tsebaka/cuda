%%cuda
#include <iostream>
#include <stdio.h>


const int N = 10;
const int blocksPerGrid = 4;
const int threadsPerBlock = 4;


__global__ void sum_exp_kernel(float* outputs, float* logits) {
    __shared__ float cache[threadsPerBlock];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float exp_sum = 0;
    for (int i = tid; i < N; i += stride) {
        exp_sum += expf(outputs[i]);
        printf("cache: &f", exp_sum);
    }
    printf("cache: &d", cache[0]);
    printf("cache: &f", exp_sum);
    cache[cacheIndex] = exp_sum;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        atomicAdd(&logits[0], cache[0]);
    }                                                     
}

__global__ void softmax_kernel(const float* outputs, float* logits) {
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        logits[i + 1] = expf(outputs[i]) / logits[0];
    }
}


int main() {
    float outputs[N], logits[N + 1];
    float *dev_outputs, *dev_logits;

    for (int i = 0; i < N; ++i) {
        outputs[i] = (expf(2 * i + 1) - expf(- 2 * i - 1)) / (expf(i + 1) + expf(- i - 1));
    }

    cudaMalloc((void**)&dev_outputs, N * sizeof(float));
    cudaMalloc((void**)&dev_logits, (N + 1) * sizeof(float));

    cudaMemcpy(dev_outputs, outputs, N * sizeof(float), cudaMemcpyHostToDevice);

    sum_exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_outputs, dev_logits);
    cudaDeviceSynchronize();

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_outputs, dev_logits);
    cudaDeviceSynchronize();

    cudaMemcpy(logits, dev_logits, (N + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_outputs);
    cudaFree(dev_logits);

    for (int i = 0; i < N + 1; ++i) {
        std::cout << logits[i] << " "; 
    }
            
    return 0;
}
