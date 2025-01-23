%%cuda
#include <iostream>
#include <stdio.h>


const int N = 10;
const int blocksPerGrid = 4;
const int threadsPerBlock = 4;

void fill(int* arr, int size) {
    for (int i = 0; i < size; ++i)
        *(arr + i) = i + 1;
}

__global__ void dot(int* a, int* b, int *dot_product) {
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

    if (cacheIndex == 0) {
        atomicAdd(dot_product, cache[0]);
    }
}

int main() {
    int a[N], b[N];
    int dot_product;
    int *dev_a, *dev_b, *dev_dot_product;
    fill(a, N);
    fill(b, N);

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_dot_product, sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_dot_product);

    cudaMemcpy(&dot_product, dev_dot_product, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_dot_product);

    std::cout << dot_product << "\n";    

    return 0;
}
