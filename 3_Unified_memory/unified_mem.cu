#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>

void copyH2D(void* dest,void* src,std::size_t size){
	cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void copyD2H(void* dest, void* src, std::size_t size) {
	cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

/*
__global__
void device_add(int *a,int *b,int *c){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N){
  	c[idx] = a[idx] + b[idx];
	}
}
*/

__global__
void device_fill_arr(float *a,float val,int N)
{
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  auto stride = blockDim.x * gridDim.x;
  for (auto i = index; i < N; i += stride)
    a[i] = val;
}

__global__
void device_add(float *a,float *b,float *c,int N)
{
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  auto stride = blockDim.x * gridDim.x;
  for (auto i = index; i < N; i += stride)
    c[i] = a[i] + b[i];
}

void fill_arr(float *data,float val,int N){
	for(int i=0;i<N;++i){
		data[i] = val;
	}
}

void print_equation(float *a,float *b,float *c, int N){
  for(int i=0;i<N;i++){
    std::cout <<a[i]<<"+"<<b[i]<<"="<<c[i]<<'\n';
  }
}


int main(void){
	constexpr unsigned int N = 1<<20;
  float *a,*b,*c;
	auto size = N * sizeof(float);
	//int N = 1<<20;

  //allocate host memory
	cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
	cudaMallocManaged(&c, size);

  fill_arr(a,1.0f,N);
  fill_arr(b,2.0f,N);

	auto blockSize = 256;
  auto numBlocks = static_cast<int>(N + blockSize - 1) / blockSize;

	//device_fill_arr<<<numBlocks,blockSize>>>(a,1.0f,N);
	//device_fill_arr<<<numBlocks,blockSize>>>(b,2.0f,N);

	//prefetch GPU memory
	int device = -1;
	cudaGetDevice(&device);
	cudaMemPrefetchAsync(a, size, device, nullptr);
	cudaMemPrefetchAsync(b, size, device, nullptr);
	cudaMemPrefetchAsync(c, size, device, nullptr);

  device_add<<<numBlocks,blockSize>>>(a,b,c,N);
	// Host prefecthes Memory
  cudaMemPrefetchAsync(c,size, cudaCpuDeviceId, nullptr);
	cudaDeviceSynchronize();
  //print_equation(a,b,c);

	// Check for errors (all values should be 3.0f)
  auto maxError = 0.0f;
  for (auto i = 0; i < N; i++)
    maxError = max(maxError, abs(c[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  //free device memory
  cudaFree(a); cudaFree(b); cudaFree(c);
  return 0;
}
