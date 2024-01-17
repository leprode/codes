#include <stdio.h>

__device__ float devData;

__device__ float inc(float* addr, float val) {
    // devData = atomicAdd(addr, val);
    // return devData;
    return atomicAdd(addr, val);
}

__global__ void checkGlobalVariable(){
    //display the original value
    printf("Device: the value of the global variable is %f\n", devData);
    //alter the value
    // devData += 2.0f;
    inc(&devData, 2.0f);
}

int main(){
    //initialize the global variable
    float value = 3.14f;
    //value to devData
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host:   copied %f to the global variable\n", value);

    //invoke the kernel
    checkGlobalVariable<<<5, 5>>>();

    //copy the global variable back to the host
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host:   the value changed by the kernel to %f\n", value);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}