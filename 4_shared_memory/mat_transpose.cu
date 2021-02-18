#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_utility.cuh"
#include <cassert>
#include "cuda_utility_func.cu"
#include <iostream>

#define N 2048
#define BLOCK_SIZE 32

__global__ void matrix_transpose_naive(int *input,int* output){
  int indexX = threadIdx.x + blockDim.x * blockIdx.x;
  int indexY = threadIdx.y + blockDim.y * blockIdx.y;

  int idx = indexY * N + indexX;
  int transposeIdx = indexX * N + indexY;
  output[idx] = input[transposeIdx];
}

__global__ void matrix_transpose_shared(int *input,int *output){

  __shared__ int sharedMemory [BLOCK_SIZE][BLOCK_SIZE+1];

  //global index
  int indexX = threadIdx.x + blockDim.x * blockIdx.x;
  int indexY = threadIdx.y + blockDim.y * blockIdx.y;
  int index = indexY * N + indexX;

  //transpose global index
  int transIdxX = threadIdx.x + blockIdx.y * blockDim.x;
  int transIdxY = threadIdx.y + blockIdx.x * blockDim.y;
  int transposeIdx = transIdxY * N + transIdxX;

  //local index
  int localIndexX = threadIdx.x;
  int localIndexY = threadIdx.y;

  sharedMemory[localIndexX][localIndexY] = input[index];

  __syncthreads();

  output[transposeIdx] = sharedMemory[localIndexY][localIndexX];
}

void fill_array(int *data){
  for(int i=0;i<N*N;i++){
    data[i] = i;
  }
}

void print_output(int *input,int *output){
  printf("Input\n");
  for(int i=0;i<N*N;i++){
    if(i%N == 0)printf("\n");
    printf(" %d ",input[i] );
  }

  printf("\nOutput\n");
  for(int i=0;i<N*N;i++){
    if(i%N == 0)printf("\n");
    printf(" %d ",output[i] );
  }
  printf("\n");
}

int main(void){
  int *a,*b;
  int *d_a,*d_b;

  int size = N * N * sizeof(int);
  a = (int *)malloc(size);fill_array(a);
  b = (int *)malloc(size);

  allocateArray((void**)&d_a,size);
  allocateArray((void**)&d_b,size);

  copyH2D(d_a,a,size);
  copyH2D(d_b,b,size);

  dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 gridSize(N/BLOCK_SIZE,N/BLOCK_SIZE,1);
  cudaEvent_t start, stop;
  float elapse;

  // initialize time
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // record initial time
  cudaEventRecord(start);

  //matrix_transpose_naive<<<gridSize,blockSize>>>(d_a,d_b);
  matrix_transpose_shared<<<gridSize,blockSize>>>(d_a,d_b);
  // record final time
  cudaEventRecord(stop);

  //wait until all events complete
  cudaEventSynchronize(stop);

  // calc
  cudaEventElapsedTime(&elapse, start, stop);
  std::cout<<"gpu :"<<elapse<<"ms"<<std::endl;

  // 終了処理
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  copyD2H(b,d_b,size);

  //print_output(a,b);

  free(a);free(b);
  freeArray(d_a);freeArray(d_b);

  return 0;
}
