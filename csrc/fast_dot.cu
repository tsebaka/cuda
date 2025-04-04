%%cuda
#include <iostream>
#include <stdio.h>


const int N = 10;
const int blocksPerGrid = 2;
const int threadsPerBlock = 4;

void fill(int* arr, int size) {
    for (int i = 0; i < size; ++i)
        *(arr + i) = i + 1;
}

__global__ void dot(int* a, int* b, int* c) {
    __shared__ int cache[threadsPerBlock];
    int cacheIndex = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int temp = 0;
    for (int i = tid; i < N; i += stride) {
        temp += a[i] * b[i];
    }
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

__global__ void sum(int *a, int *result) {
    int tid = threadIdx.x;
    int i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i)
            a[tid] += a[tid + i];
        __syncthreads();
        i /= 2;
    }
    __syncthreads();

    if (tid == 0) {
        result[tid] = a[tid];
    }
}

int main() {
    int result[1];
    int *dev_result;
    int a[N], b[N], partial_c[blocksPerGrid];
    int *dev_a, *dev_b, *dev_partial_c;
    fill(a, N);
    fill(b, N);

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    cudaDeviceSynchronize();
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(dev_partial_c, partial_c, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_result, 5 * sizeof(int));
    sum<<<blocksPerGrid, threadsPerBlock>>>(dev_partial_c, dev_result);
    cudaMemcpy(result, dev_result, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    cudaFree(dev_result);

    std::cout << result[0];

    return 0;
}
