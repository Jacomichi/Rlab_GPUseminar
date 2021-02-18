#include <stdio.h>
#include <stdlib.h>

__global__ void print_from_device(void){
	printf("Hello World! from device\n");
}

__global__ void print_from_device_w_id(void){
	printf("Hello World! from device (block : %d, threads : %d)\n",blockIdx.x,threadIdx.x);
}

int main(void){
	printf("Hello World From host!\n");

	int threads = 4;
	int blocks = 2;
	print_from_device_w_id<<<threads,blocks>>>();
	cudaDeviceSynchronize();
	return 0;
}
