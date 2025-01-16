%%cuda
#include <iostream>
#include <stdio.h>


__global__ void fast_fill(int* matrix, int rows, int cols) {
    int row = blockIdx.y;
    int col = blockIdx.x;

    if (row < rows && col < cols) {
        matrix[row * cols + col] = 1;
    }
}

int main() {
    int rows = 3, cols = 4;

    int* arr = new int[rows * cols];

    int* dev_arr;
    cudaMalloc((void**)&dev_arr, rows * cols * sizeof(int));

    cudaMemset(dev_arr, 0, rows * cols * sizeof(int));

    dim3 grid(cols, rows); 
    fast_fill<<<grid, 1>>>(dev_arr, rows, cols);

    cudaMemcpy(arr, dev_arr, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << arr[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] arr;
    cudaFree(dev_arr);

    return 0;
}
