#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_utility.cuh"
#include <cassert>
#include "cuda_utility_func.cu"
#include <iostream>

#define N 1024
#define BLOCK_SIZE 32

__global__
void matrix_mul_naive(int *A,int *B,int* output){
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if(i < N){
    for(int j = 0;j < N ; ++j){
      for(int k= 0; k < N ; ++k){
        output[i + N *j] += A[i + N* k] * B[k + N * j];
      }
    }
  }
}

__global__
void matrix_mul_naive_2d(int *A,int *B,int* output){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;

  if(i < N && j < N){
      int out = 0;
      for(int k= 0; k < N ; ++k){
        out += A[i + N* k] * B[k + N * j];
      }
      out = output[i + N *j];
  }
}


void fill_array(int *data){
  for(int i=0;i<N*N;i++){
    data[i] = i;
  }
}

void fill_array_zeros(int *data){
  for(int i=0;i<N*N;i++){
    data[i] = 0;
  }
}

void fill_eye(int *data){
  for(int i = 0; i <N;++i){
    data[i + N*i] = 1;
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
  int *a,*b,*eye;

  int size = N * N * sizeof(int);
  allocateUM(&a,size);
  allocateUM(&b,size);
  fill_array(a);
  fill_array_zeros(b);
  allocateUM(&eye,size);
  fill_eye(eye);

  //int thread_num = 1024;
  //int block_num = (N + thread_num - 1)/thread_num;

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);
  cudaEvent_t start, stop;
  float elapse;

  // initialize time
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // record initial time
  cudaEventRecord(start);

  //matrix_mul_naive<<<block_num,thread_num>>>(eye,a,b);

  matrix_mul_naive_2d<<<grid,block>>>(eye,a,b);
  // record final time
  cudaEventRecord(stop);

  //wait until all events complete
  cudaEventSynchronize(stop);

  // calc
  cudaEventElapsedTime(&elapse, start, stop);

  // 終了処理
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //print_output(a,b);
  std::cout<<"gpu :"<<elapse<<"ms"<<std::endl;


  freeArray(a);freeArray(b);freeArray(eye);

  return 0;
}
