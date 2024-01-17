#include <cstdio>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n ){
    //set thread ID
    unsigned int tid = threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    //boundary check
    if(tid>=n) return;

    //in-place reduction in global memory
    for(int stride=1;stride<blockDim.x;stride*=2){
        if((tid%(2*stride)) == 0){
            idata[tid] += idata[tid+stride];
        }

        //less time
        // for(int stride = 1;stride<blockDim.x;stride*=2){
        //     int index = 2*stride*tid;
        //     if(index<blockDim.x){
        //         idata[index]+=idata[index+stride];
        //     }
        // }

        //synchronize within block
        __syncthreads();
    }

    //write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    //two block in one thread
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x*2+threadIdx.x;

    int *idata = g_idata + blockIdx.x*blockDim.x*2;

    for(int stride = blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }

    if(tid==0) g_odata[blockIdx.x] = idata[0];
}

double cpuSecond(){
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int recursiveReduce(int *data, long const size){
    if(size == 1) return data[0];
    long const stride = size/2;
    for(long i=0; i<stride; i++){
        data[i] += data[i+stride];
    }
    return recursiveReduce(data, stride);
}

int main(int argc, char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false;

    long size = ((long)1)<<30;
    printf("with array size %ld    ",size);

    int blockSize = 512;
    if(argc>1) blockSize = atoi(argv[1]);

    dim3 block(blockSize,1);
    dim3 grid((size+block.x-1)/block.x, 1);
    printf("grid: %d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);

    size_t bytes = size*sizeof(int);
    printf("int type bytes: %d B, sum bytes: %d GB\n", sizeof(int), (size>>30) * sizeof(int));
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(grid.x*sizeof(int));
    int *tmp     = (int *)malloc(bytes);
    double iStart, iElaps;


    iStart = cpuSecond();
    for(long i=0;i<size-4;i+=4){
        h_idata[i] = (int)(rand()&0xFF);
        h_idata[i+1] = (int)(rand()&0xFF);
        h_idata[i+2] = (int)(rand()&0xFF);
        h_idata[i+3] = (int)(rand()&0xFF);
    }
    iElaps = cpuSecond()-iStart;
    printf("init data elapsed %f s\n", iElaps);
    memcpy(tmp, h_idata, bytes);


    int gpu_sum=0;

    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, grid.x*sizeof(int));

    printf("start reduce\n");

    iStart = cpuSecond();
    // int cpu_sum = 0;
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuSecond()-iStart;
    printf("cpu reduce  elapsed %f s cpu_sum:%d\n", iElaps, cpu_sum);

    iStart = cpuSecond();
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iElaps = cpuSecond()-iStart;
    printf("copy data to device elapsed %f s\n", iElaps);

    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond()-iStart;
    printf("gpu reduce  elapsed %f s\n", iElaps);

    iStart = cpuSecond();    
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    iElaps = cpuSecond()-iStart;
    printf("copy data back elapsed %f s\n", iElaps);

    iStart = cpuSecond();
    gpu_sum=0;
    for(int i=0;i<grid.x;i++){
        gpu_sum+=h_odata[i];
    }
    iElaps = cpuSecond()-iStart;
    printf("gpu reduce postprecess elapsed %f s gpu_sum:%d\n", iElaps, gpu_sum);
    

    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();

    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");
    else printf("Test succeed!\n");
    return 0;
}