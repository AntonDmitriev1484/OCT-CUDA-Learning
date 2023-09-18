#include "stdio.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n"); // Doesn't actually run - shouldn't
}

int main() {
    printf("Hello World from CPU!\n");
    cuda_hello<<<1,1>>>();
    return 0;
}