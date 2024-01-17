#include <stdio.h>

__global__ void hello_world()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;  
    printf("Hello World from gpu, my thread id is (%d, %d, %d), my block id is (%d, %d, %d), my sum block id is (%d, %d, %d), my global id is %d\n", 
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, 
        blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z, 
        blockId*blockDim.x*blockDim.y*blockDim.z + threadIdx.x*blockDim.y*blockDim.z + threadIdx.y*blockDim.z + threadIdx.z);
}

int main()
{
    printf("Hello World from cpu!\n");
    dim3 grid = dim3(2, 2, 2);
    dim3 block = dim3(1, 2, 3);
    printf("my grid dim is (%d, %d, %d), my block dim is (%d, %d, %d)\n", 
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z);
    hello_world<<<grid, block>>>();
    printf("*****************\n");
    // cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}

