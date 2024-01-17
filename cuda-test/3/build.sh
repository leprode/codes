echo "building simpleDivergence.cu..."
nvcc simpleDivergence.cu -o simpleDivergence

echo "building simpleDivergence.cu with o1..."
nvcc -O1 simpleDivergence.cu -o simpleDivergence-O1

echo "building simpleDivergence.cu with o3..."
nvcc -O3 simpleDivergence.cu -o simpleDivergence-O3

echo "building reduceNeighbored.cu..."
nvcc reduceNeighbored.cu -o reduceNeighbored

echo "building nestedhelloworld.cu..."
nvcc -rdc=true nestedhelloworld.cu -o nestedhelloworld -lcudadevrt