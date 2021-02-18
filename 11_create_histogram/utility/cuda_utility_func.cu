#include <cuda.h>
#include "cuda_utility.cuh"
#include <cassert>

#ifndef _CUDA_UTIL_FUNC_

void allocateArray(void **devPtr, size_t size)
{
	CudaSafeCall(cudaMalloc(devPtr, size));
}

template < typename T >
void allocateUM(T *devPtr, size_t size)
{
	CudaSafeCall(cudaMallocManaged(devPtr, size));
}

void freeArray(void *devPtr)
{
	CudaSafeCall(cudaFree(devPtr));
}

void threadSync()
{
	CudaSafeCall(cudaThreadSynchronize());
}

void copyD2H(void* dst, void* src, size_t memSize) {
	CudaSafeCall(cudaMemcpy(dst, src, memSize, cudaMemcpyDeviceToHost));
}

void copyH2D(void* dst, void* src, size_t memSize) {
	CudaSafeCall(cudaMemcpy(dst, src, memSize, cudaMemcpyHostToDevice));
}


#define copyH2S(symbol, src, memSize) \
	CudaSafeCall(cudaMemcpyToSymbol(symbol, src, memSize))


#endif
