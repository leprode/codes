#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

void get_format_time_ms(char *str_time) { 
    struct tm *tm_t;
    struct timeval time;
    
    gettimeofday(&time,NULL);
    tm_t = localtime(&time.tv_sec);
    if(NULL != tm_t) {
        sprintf(str_time,"%04d-%02d-%02d %02d:%02d:%02d.%03ld",
            tm_t->tm_year+1900,
            tm_t->tm_mon+1, 
            tm_t->tm_mday,
            tm_t->tm_hour, 
            tm_t->tm_min, 
            tm_t->tm_sec,
            time.tv_usec/1000);
    }
    return;
}


int main(int argc, char* argv[])
{
  int size = 32*1024*1024;


  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  printf("[MPI Rank %d] my hostname: %s \n", myRank, hostname);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }


  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);

  printf("[MPI Rank %d] after get uniqueID \n", myRank);

  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  printf("[MPI Rank %d] after MPI_Bcast\n", myRank);
  char line[1000];
  get_format_time_ms(line);
  printf("[MPI Rank %d] now time: %s\n", myRank, line);
  sleep(1);
  get_format_time_ms(line);
  printf("[MPI Rank %d] now time: %s\n", myRank, line);

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));

  printf("[MPI Rank %d] after cudaStreamCreate\n", myRank);
  // sleep(500);

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  printf("[MPI Rank %d] after nccl comm init rank\n", myRank);
  // sleep(500);

  struct timespec startT, endT;
  if( myRank == 0) {
    get_format_time_ms(line);
    printf("[MPI Rank %d] now time: %s\n", myRank, line);

    clock_gettime(CLOCK_MONOTONIC, &startT);
  }

  ncclGroupStart();
  //communicating using NCCL
  for(int i=0; i<1000; i++) {
    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));
  }
  ncclGroupEnd();
  if( myRank == 0) {
    clock_gettime(CLOCK_MONOTONIC, &endT);
    uint64_t delta_us = (endT.tv_sec - startT.tv_sec) * 1000000 + (endT.tv_nsec - startT.tv_nsec) / 1000;
    // uint32_t delta_ms = (endT.tv_sec - startT.tv_sec) * 1000 + (endT.tv_nsec - startT.tv_nsec) / 1000000;
    get_format_time_ms(line);
    printf("[MPI Rank %d] after all reduce, now time: %s, cost time: %lu us\n", myRank, line, delta_us);
  }

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));


  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}