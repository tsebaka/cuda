%%cuda
#include <iostream>
#include <stdio.h>


__global__ void mat_mul_kernel(float* mat_a, float* mat_b, size_t rows, size_t cols, float* result) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    printf("ThreadIdx: (%d, %d) | BlockIdx: (%d, %d)\n", tx, ty, bx, by);
    printf("row, col: (%d, %d)\n", row, col);

    float _sum = 0;
    for (int k = 0; k < cols; ++k) {
        _sum += mat_a[cols * row + k] * mat_b[k * cols + col];   
    }
    result[row * cols + col] = _sum;
}


int main() {
    size_t rows = 3;
    size_t cols = 3;
    float mat_a[rows * cols] = {
        1, 1, -1,
        1, -1, 1,
        -1, 1, 1
    };
    float mat_b[rows * cols] = {
        0.5, 0.5, 0,
        0.5, 0, 0.5,
        0, 0.5, 0.5
    };
    float result[rows * cols] = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };
    float* dev_mat_a;
    float* dev_mat_b;
    float* dev_result;

    cudaMalloc((void**)&dev_mat_a, rows * cols * sizeof(float));
    cudaMalloc((void**)&dev_mat_b, rows * cols * sizeof(float));
    cudaMalloc((void**)&dev_result, rows * cols * sizeof(float));

    cudaMemcpy(dev_mat_a, mat_a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mat_b, mat_b, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(1, 1);
    dim3 threadPerBlock(rows, cols);
    mat_mul_kernel<<<blocksPerGrid, threadPerBlock>>>(dev_mat_a, dev_mat_b, rows, cols, dev_result);

    cudaMemcpy(result, dev_result, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows * cols; ++i) {
        std::cout << result[i] << " "; 
    }

    cudaFree(dev_mat_a);
    cudaFree(dev_mat_b);
    cudaFree(dev_result);
    
    return 0;
}
