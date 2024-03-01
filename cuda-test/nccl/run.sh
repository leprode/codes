# export NCCL_DEBUG=TRACE 
# export NCCL_DEBUG_SUBSYS=ALL 
# export NCCL_SOCKET_IFNAME=enp94s0f0
# export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=3,4
mpirun -np 2 ./example2