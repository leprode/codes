#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
    CUresult code=cuInit(0);

    int driverVersion;
    int runtimeVersion;

    // 获取CUDA驱动版本
    CUresult res = cuDriverGetVersion(&driverVersion);
    if (res != CUDA_SUCCESS) {
        printf("Error getting CUDA driver version: %d\n", res);
    } else {
        printf("CUDA Driver Version: %d\n", driverVersion);
    }

    // 获取CUDA运行时版本
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        printf("Error getting CUDA runtime version: %s\n", cudaGetErrorString(err));
    } else {
        printf("CUDA Runtime Version: %d\n", runtimeVersion);
    }

    return 0;
}