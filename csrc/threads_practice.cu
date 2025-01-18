#include <iostream>
#include <stdio.h>

#define N 100


__global__ void fast_sum(int* a, int* b, int* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = N - i;
    }

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    fast_sum<<<1, N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < N; ++i) {
        std::cout << c[i] << " ";
    }


    return 0;
}
