#!/bin/bash

# Configuration
# ~ --gpu-architecture=compute_50 --gpu-code=sm_50,sm_52
CUDA_GENCODE="-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_50,code=sm_50"


cd src
# add nvcc path to .bashrc, better for python build.py
CUDA_HOME="/nfs/xs/local/cuda-10.1/"
${CUDA_HOME}bin/nvcc -I${CUDA_HOME}/include --expt-extended-lambda -O3 -c bn.cu -o bn.o -x cu -Xcompiler -fPIC -std=c++11 ${CUDA_GENCODE}
cd ..
