#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 1024

void copyH2D(void* dest,void* src,std::size_t size){
	cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void copyD2H(void* dest, void* src, std::size_t size) {
	cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

__global__ void device_add(int *a,int *b,int *c){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N){
  	c[idx] = a[idx] + b[idx];
	}
}

void fill_arr(int *data,int val){
	for(int i=0;i<N;++i){
		data[i] = val;
	}
}

void print_equation(int *a,int *b,int *c){
  for(int i=0;i<N;i++){
    std::cout <<a[i]<<"+"<<b[i]<<"="<<c[i]<<'\n';
  }
}


int main(void){
  int *a,*b,*c;
  int *d_a,*d_b,*d_c;
  int int_size = N * sizeof(int);

  //allocate host memory
	/*
	a = (int *)malloc(int_size);
	fill_arr(a,1);
	b = (int *)malloc(int_size);
	fill_arr(b,2);
	c = (int *)malloc(int_size);
	*/

  cudaMallocHost(&a,int_size);
  fill_arr(a,1);
  cudaMallocHost(&b,int_size);
  fill_arr(b,2);
  cudaMallocHost(&c,int_size);

  //allocate device memory
	cudaMalloc(&d_a,int_size);
  cudaMalloc(&d_b,int_size);
  cudaMalloc(&d_c,int_size);

  copyH2D(d_a, a, int_size);
  copyH2D(d_b, b, int_size);

  int blocks = 8;
	//castしないとN/blocksの段階で小数点以下が落とされる。
  int threads = std::ceil(static_cast<double>(N)/blocks);

  device_add<<<blocks,threads>>>(d_a,d_b,d_c);
  copyD2H(c, d_c, int_size);
  print_equation(a,b,c);

  //free host memory
  cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(c);
	//free(a);free(b);free(c);

  //free device memory
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
