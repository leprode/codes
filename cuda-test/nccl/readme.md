参考[链接](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html)

# 编译
./build.sh

# 运行
./singleProcess
./example2
mpirun -np 4  example2

# nccl日志全开
export NCCL_DEBUG=TRACE  # INFO
export NCCL_DEBUG_SUBSYS=ALL