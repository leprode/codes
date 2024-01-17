#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mathKernel1(float *c){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float a, b;
    a = b = 1.0f;

    if(tid%2==0){//线程级编程，会导致线程束分化
                //CUDA编译器会自动优化
        a = 100.0f;
    }else{
        b = 200.0f;
    }

    for (int i=0; i<100; i++){
        c[tid] += a*b;
    }
}

__global__ void mathKernel2(float *c){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float a, b;
    a = b = 1.0f;

    if((tid/warpSize)%2==0){//使分支粒度为线程束大小倍数
        a = 100.0f;
    }else{
        b = 200.0f;
    }
    for (int i=0; i<100; i++){
        c[tid] += a*b;
    }
}

__global__ void mathKernel3(float *c){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float a, b;
    a = b = 1.0f;

    bool ipred = tid%2==0;
    if(ipred){//使分支粒度为线程束大小倍数
        a = 100.0f;
    }
    if (!ipred){
        b = 200.0f;
    }
    for (int i=0; i<100; i++){
        c[tid] += a*b;
    }
    
}


void sumArraysOnHost(float *C, const int N){
    float a, b;
    a = b = 1.0f;

    for (int tid=0; tid<N; tid++){
        if(tid%2==0){
            a = 100.0f;
        }else{
            b = 200.0f;
        }
        for (int i=0; i<100; i++){
            C[tid] += a*b;
        }
    }
    
}

void sumArraysOnHost4(float *C, const int N){
    float a, b;
    a = b = 1.0f;

    for (int tid=0; tid<N; tid+=4){
        float a0, a1, a2, a3;
        float b0, b1, b2, b3;
        a0 = a2 = 100.0f;
        a1 = a3 = a;
        b0 = b2 = b;
        b1 = b3 = 200.0f;
        for (int i=0; i<100; i++){
            C[tid] += a0*b0;
            C[tid+1] += a1*b1;
            C[tid+2] += a2*b2;
            C[tid+3] += a3*b3;
        }
    }
    
}


double cpuSecond(){
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    int size = 1<<24;
    int blocksize = 64;
    printf("size:%d, blocksize:%d\n", size, blocksize);

    dim3 block (blocksize, 1);
    dim3 grid ((size + blocksize - 1)/blocksize, 1);
    printf("grid.x:%d, grid.y:%d\n", grid.x, grid.y);
    printf("block.x:%d, block.y:%d\n", block.x, block.y);

    float *d_c;
    cudaMalloc((void**)&d_c, size*sizeof(float));

    double iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<<%d, %d>>>: %f s\n", grid.x, block.x,iElaps);

    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<<%d, %d>>>: %f s\n", grid.x, block.x,iElaps);

    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<<%d, %d>>>: %f s\n", grid.x, block.x,iElaps);

    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(d_c);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3 <<<%d, %d>>>: %f s\n", grid.x, block.x,iElaps);

    float *h_c;
    h_c = (float *)malloc(size*sizeof(float));

    iStart = cpuSecond();
    sumArraysOnHost(h_c, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("cpu: %f s\n",iElaps);

    iStart = cpuSecond();
    sumArraysOnHost4(h_c, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("cpu4: %f s\n",iElaps);
    return 0;
}