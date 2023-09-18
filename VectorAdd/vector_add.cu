#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compile & Run
// nvcc hello.cu -o hello
// nvcc -g -o vadd vector_add.cu
// ./hello

__global__ void vector_add(int* v1, int* v2, int* v3){
    v3[threadIdx.x] = v1[threadIdx.x] + v2[threadIdx.x];
}

int main() {
    printf("Adding Vectors!\n");

    int *v1, *v2, *v3; // Pointers to the base of our arrays
    int *gpu_v1, *gpu_v2, *gpu_v3; // Pointer to where corresponding arrays will live on GPU

    const int N = 8;
    const int num_bytes = sizeof(int) * N;

    // First allocate memory on Host
    v1 = (int*) malloc(num_bytes);
    for (int i = 0; i< N; i++) { // Initialize array values
        v1[i] = 10;
        printf("%d, ", v1[i]);
    }
    printf("\n+\n");
    v2 = (int*) malloc(num_bytes);
    for (int i = 0; i< N; i++) {
        v2[i] = i;
        printf("%d, ", v2[i]);
    }
    printf("\n");

    v3 = (int*) malloc(num_bytes);

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
    dim3 grid(1,1,1);
    dim3 vector(N,1,1);
    vector_add<<<grid, vector>>>(gpu_v1,gpu_v2,gpu_v3);

    cudaMemcpy(v3, gpu_v3, num_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i< N; i++) {
        printf("%d, ", v3[i]);
    }


    return 0;
}