#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <time.h>

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0;i<size;i++){
        ip[i] = (float)( rand() & 0xFF)/10.0f;
    }
}

void printArray(float *A, int size){
    printf("[");
    for(int i=0;i<size;i++){
        printf("%f ", A[i]);
    }
    printf("]\n");
}

double cpuSecond(){
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for(int idx=0;idx<N;idx++){
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv){
    
    int nElem = 1<<26;
    size_t nBytes = nElem * sizeof(float);

    int iLen = 256;
    dim3 block (iLen);
    dim3 grid ((nElem + iLen - 1) / iLen);

    double iStart, iElaps;
    // double iStartOnlyCal, iElapsOnlyCal, iElapsOnlyCpy, iCpy;

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, nBytes);
    cudaMalloc((void **)&d_B, nBytes);
    cudaMalloc((void **)&d_C, nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // iStart = cpuSecond();
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    // iElapsOnlyCpy = cpuSecond() - iStart;
    // iStartOnlyCal = cpuSecond();
    sumArraysOnGPU<<<grid,block>>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    // iElapsOnlyCal = cpuSecond() - iStartOnlyCal;
    // iCpy = cpuSecond();
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    // iElapsOnlyCpy = iElapsOnlyCpy + cpuSecond() - iCpy;
    // iElaps = cpuSecond() - iStart;
    // printf("Elapsed time in GPU: %f s, cal time: %f s, cpy time: %f s, sumArraysOnGPU<<<%d,%d>>>\n", 
    //     iElaps, iElapsOnlyCal, iElapsOnlyCpy,
    //     grid.x, block.x);

    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    iElaps = cpuSecond() - iStart;
    printf("Elapsed time in CPU: %f s\n", iElaps);

    if (nElem <= 10) {
        printArray(h_A, nElem);
        printArray(h_B, nElem);
        printArray(h_C, nElem);
    }


    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}