echo "building globalVariable.cu..."
nvcc globalVariable.cu -o globalVariable

echo "building memTransfer.cu..."
nvcc memTransfer.cu -o memTransfer

echo "building transpose.cu..."
nvcc transpose.cu -o transpose

