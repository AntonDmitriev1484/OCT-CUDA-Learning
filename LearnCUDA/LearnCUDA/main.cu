#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Compile & Run
// nvcc hello.cu -o hello
// nvcc -g -o vadd vector_add.cu
// ./hello

__global__ void vector_add_kernel(int* v1, int* v2, int* v3) {
    v3[threadIdx.x] = v1[threadIdx.x] + v2[threadIdx.x];
}

__global__ void vector_cross_kernel(int* v1, int* v2, int* out) {
    *out += v1[threadIdx.x] + v2[threadIdx.x];
}

void vector_add() {
    printf("Adding Vectors!\n");

    int* v1, * v2, * v3; // Pointers to the base of our arrays
    int* gpu_v1, * gpu_v2, * gpu_v3; // Pointer to where corresponding arrays will live on GPU

    const int N = 8;
    const int num_bytes = sizeof(int) * N;

    // First allocate memory on Host
    v1 = (int*)malloc(num_bytes);
    for (int i = 0; i < N; i++) { // Initialize array values
        v1[i] = 10;
        printf("%d, ", v1[i]);
    }
    printf("\n+\n");
    v2 = (int*)malloc(num_bytes);
    for (int i = 0; i < N; i++) {
        v2[i] = i;
        printf("%d, ", v2[i]);
    }
    printf("\n");

    v3 = (int*)malloc(num_bytes);

    // Then transfer to GPU
    cudaMalloc((void**)&gpu_v1, num_bytes);
    // Set gpu_v1 pointer to point to the base of this newly allocated block in GPU global memory
    cudaMemcpy(gpu_v1, v1, num_bytes, cudaMemcpyHostToDevice);
    // Copy to 1. GPU Dest Pointer from 2. Host Pointer 3. this number of bytes

    cudaMalloc((void**)&gpu_v2, num_bytes);
    cudaMemcpy(gpu_v2, v2, num_bytes, cudaMemcpyHostToDevice); // memcpy is always dst pointer, src pointer

    // CUDA malloc returns an error value. If successful, updates the pointer passed in.
    cudaError err = cudaMalloc((void**)&gpu_v3, num_bytes);
    if (err) {
        printf("Error!: %s\n", cudaGetErrorString(err));
    }
    //cudaMemcpy(gpu_v3, v3, num_bytes); // memcpy is always src pointer, dst pointer
    // v3 is currently uninitialized, so there's no point in memcopying it

    // Want 1D block, thread of size N -> 1 thread per index
    dim3 grid(1, 1, 1);
    dim3 block(N, 1, 1);
    vector_add_kernel << <grid, block >> > (gpu_v1, gpu_v2, gpu_v3);

    cudaMemcpy(v3, gpu_v3, num_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%d, ", v3[i]);
    }

    //free()
}


void vector_cross() {
    int* v1, * v2, * out; // Pointers to the base of our arrays
    int* gpu_v1, * gpu_v2, * gpu_out; // Pointer to where corresponding arrays will live on GPU

    const int N = 8;
    const int num_bytes = sizeof(int) * N;

    v1 = (int*)malloc(num_bytes);
    for (int i = 0; i < N; i++) {
        v1[i] = 10;
        printf("%d, ", v1[i]);
    }
    printf("\n+\n");
    v2 = (int*)malloc(num_bytes);
    for (int i = 0; i < N; i++) {
        v2[i] = i;
        printf("%d, ", v2[i]);
    }
    printf("\n");

    out = (int*)malloc(sizeof(int));

    cudaMalloc((void**)&gpu_v1, num_bytes);
    cudaMemcpy(gpu_v1, v1, num_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_v2, num_bytes);
    cudaMemcpy(gpu_v2, v2, num_bytes, cudaMemcpyHostToDevice);

    cudaError err = cudaMalloc((void**)&gpu_out, num_bytes);
    if (err) {
        printf("Error!: %s\n", cudaGetErrorString(err));
    }

    dim3 grid(1, 1, 1);
    dim3 block(N, 1, 1);
    vector_cross_kernel << <grid, block >> > (gpu_v1, gpu_v2, gpu_out);

    cudaMemcpy(out, gpu_out, sizeof(int), cudaMemcpyDeviceToHost); // Because we're copying a single number out

    printf(" v1 X v2 = %d ", out);

    // Without synchronization over the single memory value, this unsurprisingly creates a race condition!
}

__global__ void matrix_mult_kernel(int* m1, int* m2, int P, int* m3) {

    // Note: Our block has the same dimensions as m3

    //const int access_m3_at_x_y = threadIdx.x * (blockDim.y) + threadIdx.y;

    //for (int p = 0; p < P; p++) {
    //    const int access_m1_at_x_p = threadIdx.x * (blockDim.y) + p;
    //    const int access_m2_at_p_y = p * (blockDim.y) + threadIdx.y;

    //    m3[0] += m1[access_m1_at_x_p] * m2[access_m2_at_p_y];
    //}\

    // Yeah - I guess the problem was how I was calculating the index
    // I was using the right formula, just the wrong variables?
    // Why block instead of thread if there is only 1 block????

    int ROW = blockIdx.y * blockDim.y + threadIdx.y; // didn't do the formulas all the way
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    int tmpSum = 0;

    for (int i = 0; i < P; i++) {
        tmpSum += m1[ROW * P + i] * m2[i * P + COL];
    }

    m3[ROW * P + COL] = tmpSum;
}

void simple_matrix_mult() {
    // m1 is an M x P matrix
    // m2 is a P x N matrix
    // m3 will be a M x N matrix

    const int P = 3;
    const int M = 2;
    const int N = 2;

    int* m1, * m2, * m3;
    int* gpu_m1, * gpu_m2, * gpu_m3;


    // Allocating m1 as a 1D array -> we will index it as a 2D array in our kernel
    const int m1_size = M * P * sizeof(int);
    m1 = (int*)malloc(m1_size);
    for (int i = 0; i < M * P; i++) {
            m1[i] = 1;
            printf("%d ", m1[i]);
    }
    cudaMalloc((void**)&gpu_m1, m1_size);
    cudaMemcpy(gpu_m1, m1, m1_size, cudaMemcpyHostToDevice);

    printf("\n");

    const int m2_size = P * N * sizeof(int);
    m2 = (int*)malloc(m2_size);
    for (int i = 0; i < P * N; i++) {
        m2[i] = 1;
        printf("%d ", m2[i]);
    }
    cudaMalloc((void**)&gpu_m2, m2_size);
    cudaMemcpy(gpu_m2, m2, m2_size, cudaMemcpyHostToDevice);

    const int m3_size = M * N * sizeof(int);
    m3 = (int*)malloc(m3_size);
    cudaError err = cudaMalloc((void**)&gpu_m3, m3_size);
    if (err) {
        printf("Error!: %s\n", cudaGetErrorString(err));
    }
    for (int i = 0; i < M * N; i++) { // Initialize m3
        m3[i] = 0;
    }
    cudaMemcpy(gpu_m3, m3, m3_size, cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 block(M, N, 1);

    matrix_mult_kernel <<<grid, block>>> (gpu_m1, gpu_m2, P, gpu_m3);
    cudaMemcpy(m3, gpu_m3, m3_size, cudaMemcpyDeviceToHost); // Commenting this out didnt change anything so it must be that memory on the GPU isn't being initialized

    printf("\n");
    for (int i = 0; i < M; i++) {
        
        for (int j = 0; j < N; j++) {
            printf("%d ", m3[i * N + j]);
        }
        printf("\n");
    }

    // Ok, we're getting  -842150448 in every slot, I think this might be because memory isn't being copied back correctly?
    // Its either
    // 1. Not updating the gpu memory -> gpu memory remains unchanged -> that unchanged memory gets copied over
    // 2. Not copying properly, gpu memory gets changed -> not copied into m3.
}

int main() {
    simple_matrix_mult();

    return 0;
}