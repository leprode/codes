# INCLUDEPATH += /usr/local/cuda/include
# LIBS += -L/usr/local/cuda/lib64
# LIBS += -lcuda  #要添加这个哦，不然会出现error: undefined reference to `cuInit'的错误

nvcc --device-c version.cu 
nvcc version.o -L/usr/local/cuda/lib64 -lcuda -o version 

# nvcc version.cu -o version