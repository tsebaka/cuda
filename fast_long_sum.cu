#include <iostream>
#include <stdio.h>

#define N 10


void fill(int* arr, int size) {
    for (int i = 0; i < size; ++i)
        *(arr + i) = i + 1;
}

__global__ void sum(int* a, int* b, int* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        printf("Thread ID after: %d\n", tid);
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
        printf("Thread ID before: %d\n", tid);
    }
}

int main() {
    int a[N], b[N], c[N];
    int* dev_a, *dev_b, *dev_c;
    fill(a, N);
    fill(b, N);

    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice);

    sum<<<2, 2>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i)
        std::cout << c[i] << " ";     
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
