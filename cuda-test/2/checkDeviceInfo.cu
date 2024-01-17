#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA device count: %d\n", deviceCount);

    cudaDeviceProp deviceProp;
    for (int i = 0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device %d: \"%s\"\n", i, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %lu\n", deviceProp.totalGlobalMem);
        printf("  Max Texture Dimension Size: 1D=(%d), 2D=(%d,%d)\n", deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
        printf("  Max Layered 1D Texture Size: (num)=%d, size=%d\n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Max Layered 2D Texture Size: (num)=%d, size=(%d,%d)\n", deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
        printf("  Total amount of constant memory: %lu\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block: %lu\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize);
        printf("  Maximum number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  GPU Clock rate: %f MHz\n", deviceProp.clockRate * 1e-3);
        printf("  Memory Clock rate: %f MHz\n", deviceProp.memoryClockRate * 1e-3);
        printf("  Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
        printf("  L2 Cache Size: %lu\n", deviceProp.l2CacheSize);
        printf("  Max Texture 2D (width, height) in pixels: (%d, %d)\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
        printf("  Max Texture 1D (width) in pixels: %d\n", deviceProp.maxTexture1D);
        printf("  Max Texture 1D Layered (width) in pixels: %d\n", deviceProp.maxTexture1DLayered[1]);
        printf("  Max Texture 2D Layered (width, height) in pixels: (%d, %d)\n", deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1]);
        printf("  Total amount of constant memory: %lu\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block: %lu\n", deviceProp.sharedMemPerBlock);
        printf("  Total amount of shared memory per multiprocessor: %lu\n", deviceProp.sharedMemPerMultiprocessor);
        printf("  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Kernel execution timeout: %s\n", deviceProp.kernelExecTimeoutEnabled? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory: %s\n", deviceProp.integrated? "Yes" : "No");
        printf("  Support host page-locked memory mapping: %s\n", deviceProp.canMapHostMemory? "Yes" : "No");
        printf("  Alignment requirement for Surfaces: %s\n", deviceProp.surfaceAlignment? "Yes" : "No");
        printf("  Device has ECC support: %s\n", deviceProp.ECCEnabled? "Yes" : "No");
        printf("  Device supports Unified Addressing (UVA): %s\n", deviceProp.unifiedAddressing? "Yes" : "No");
        printf("  Device PCI Bus ID / PCI location ID: %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);
    }
    return 0;
}