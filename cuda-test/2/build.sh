# nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o sum
echo "building sumArraysOnHost.c..."
nvcc sumArraysOnHost.c -o sum

echo "building sumArraysOnGPU.cu..."
nvcc sumArraysOnGPU.cu -o sumOnGPU

echo "building sumMatrixOnGPU-2D-grid-2D-block.cu..."
nvcc sumMatrixOnGPU-2D-grid-2D-block.cu -o sumMatrixOnGPU-2D-grid-2D-block

echo "building checkDeviceInfo.cu..."
nvcc checkDeviceInfo.cu -o checkDeviceInfo