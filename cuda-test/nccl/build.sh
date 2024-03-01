echo "building singleProcess.cu..."
nvcc singleProcess.cu -o singleProcess -I /usr/local/nccl/include -L /usr/local/nccl/lib -l nccl

echo "building example2.c..."
CUDA_INSTALL_PATH=/usr/local/cuda-12.1
mpic++ example2.c -o example2 -I /usr/local/nccl/include -L /usr/local/nccl/lib -l nccl -l cudart -l curand -L ${CUDA_INSTALL_PATH}/lib64

#### 10.101.116.18
# NCCL_INSTALL_PATH=/usr/local/nccl_2.5.6
# echo "building singleProcess.cu..."
# nvcc singleProcess.cu -o singleProcess -I ${NCCL_INSTALL_PATH}/include -L ${NCCL_INSTALL_PATH}/lib -l nccl

# echo "building example2.c..."
# CUDA_INSTALL_PATH=/usr/local/cuda-12.1
# MPI_HOME=/data/home/wangxj/workspace/lib/openmpi-5.0.2
# #export LD_LIBRARY_PATH="${CUDA_INSTALL_PATH}/targets/x86_64-linux/lib:${CUDA_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH" 
# nvcc example2.c -o example2 -I ${NCCL_INSTALL_PATH}/include -L ${NCCL_INSTALL_PATH}/lib -l nccl -l cudart -l curand -L ${MPI_HOME}/lib -L ${MPI_HOME}/lib64 -lmpi

#### 10.101.116.17
# NCCL_INSTALL_PATH=/usr/local/nccl_2.5.6
# echo "building singleProcess.cu..."
# nvcc singleProcess.cu -o singleProcess -I ${NCCL_INSTALL_PATH}/include -L ${NCCL_INSTALL_PATH}/lib -l nccl

# echo "building example2.c..."
# CUDA_INSTALL_PATH=/usr/local/cuda-12.1
# MPI_HOME=/data/home/wangxj/workspace/lib/openmpi-5.0.2
# #export LD_LIBRARY_PATH="${CUDA_INSTALL_PATH}/targets/x86_64-linux/lib:${CUDA_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH" 
# nvcc example2.c -o example2 -I ${NCCL_INSTALL_PATH}/include -L ${NCCL_INSTALL_PATH}/lib -l nccl -l cudart -l curand -L ${MPI_HOME}/lib -L ${MPI_HOME}/lib64 -lmpi