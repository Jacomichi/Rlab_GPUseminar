#include <stdio.h>
#include <stdlib.h>
#include <iostream>

__global__
void add_vec(int *a,int *b, int offset,int N)
{
  	int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
	if(i < N){
		a[i] = a[i] + b[i];
	}
}

template <typename T>
void fill_arr(T *data,T val,int N){
	for(int i=0;i<N;++i){
		data[i] = val;
	}
}

int main(){
    int N = 1<<28;
    int *a,*b,*d_a,*d_b;
	int int_size = N * sizeof(int);

  	//allocate host memory
  	a = (int *)malloc(int_size);
  	fill_arr(a,1,N);
  	b = (int *)malloc(int_size);
  	fill_arr(b,2,N);
	cudaMalloc((void **)&d_a,int_size);
  	cudaMalloc((void **)&d_b,int_size);

	int threads = 256;
	int blocks = 16;

	cudaStream_t *i_stream;
	int num_stream = 16;
	int StreamSize = N/num_stream;
	size_t StreamBytes = StreamSize * sizeof(float);

    i_stream = (cudaStream_t*) new cudaStream_t[num_stream];

    for (int i = 0; i < num_stream; i++){
        cudaStreamCreate(&i_stream[i]);
    }

    for (int i = 0; i < num_stream; i++){
        int offset = i * StreamSize;
        cudaMemcpyAsync(&d_a[offset],&a[offset],StreamBytes,cudaMemcpyHostToDevice,i_stream[i]);
        cudaMemcpyAsync(&d_b[offset],&b[offset],StreamBytes,cudaMemcpyHostToDevice,i_stream[i]);
        add_vec<<< threads, blocks, 0, i_stream[i] >>>(d_a,d_b,offset,N);
		cudaMemcpyAsync(&a[offset],&d_a[offset],StreamBytes,cudaMemcpyDeviceToHost,i_stream[i]);
    }

	for (int i = 0; i < num_stream; i++){
        int offset = i * StreamSize;
		cudaMemcpyAsync(&a[offset],&d_a[offset],StreamBytes,cudaMemcpyDeviceToHost,i_stream[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < num_stream; i++){
        cudaStreamDestroy(i_stream[i]);
    }

    delete [] i_stream;

	free(a); free(b); cudaFree(d_a); cudaFree(d_b);
}
