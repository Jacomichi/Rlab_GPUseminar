#ifndef REDUCTION_CUH
#define REDUCTION_CUH

__global__
void reduction_kernel(double *output,double *input, int N){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.x;

    extern __shared__ double shmem[];
    shmem[i] = (idx < N) ? input[idx] : 0.0;

    __syncthreads();


    for(int stride = 1;stride < blockDim.x; stride *= 2){
        int index = 2 * stride * i;
        if(index < blockDim.x){
            shmem[i] += shmem[i + stride];
        }
        __syncthreads();
    }

    if(i == 0){
        output[blockIdx.x] = shmem[0];
    }
}

void reduction(double *output,double *input, int num_threads, int arr_size){
    int size = arr_size;
    while(size > 1){
        int num_blocks = (size + num_threads - 1)/num_threads;
        reduction_kernel<<<num_blocks,num_threads,num_threads * sizeof(double),0>>>(output, input, size);
        size = num_blocks;
    }
}

#endif
