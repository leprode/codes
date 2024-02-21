echo "building singleProcess.cu..."
nvcc singleProcess.cu -o singleProcess -I /usr/local/nccl/include -L /usr/local/nccl/lib -l nccl

echo "building example2.c..."
CUDA_INSTALL_PATH=/usr/local/cuda-12.1
mpic++ example2.c -o example2 -I /usr/local/nccl/include -L /usr/local/nccl/lib -l nccl -l cudart -l curand -L ${CUDA_INSTALL_PATH}/lib64

