#ifndef __CUDA_UTIL_H_
#define __CUDA_UTIL_H_

#include <cuda.h>
#include <stdio.h>

#ifndef DEVICE
#define DEVICE 0 
#endif

#define MAX_BLOCKS_PER_SM 2 

inline void cudaCheckError(char* file, int line, cudaError_t ce)
{
        if (ce != cudaSuccess) {
                printf("Error: file %s, line %d, %s\n", file, line, cudaGetErrorString(ce));
                exit(1);
        }
}

/* initializes the device and the memory pool */
inline void init_device(int device){
	int *h_p = (int*)malloc(1024);
        int *d_p;
        cudaSetDevice(device);
        cudaMalloc((void**)&d_p,1024);
        cudaMemcpy( d_p, h_p, 1024, cudaMemcpyHostToDevice);
        free(h_p);
        cudaFree(d_p);
        printf("Device %d initialized\n",device);
}

#endif
