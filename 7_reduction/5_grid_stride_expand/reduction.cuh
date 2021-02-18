#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <algorithm>

#define NUM_LOAD 4

__global__
void reduction_kernel(double *output,double *input, int N){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.x;

    extern __shared__ double shmem[];
    double local_sum[NUM_LOAD] = {0.0};
    for(int j = idx; j < N ; j += blockDim.x * gridDim.x*NUM_LOAD){
        for(int k = 0 ; k < NUM_LOAD;++k){
            local_sum[k] += (j + k * blockDim.x * gridDim.x < N) ? input[j + k * blockDim.x * gridDim.x] : 0.0;
        }
    }

    for(int j = 1; j< NUM_LOAD; ++j){
        local_sum[0] += local_sum[j];
    }

    shmem[i] = local_sum[0];

    __syncthreads();


    for(int stride = blockDim.x/2;stride > 0; stride >>= 1){
        if(i < stride){
            shmem[i] += shmem[i + stride];
        }
        __syncthreads();
    }

    if(i == 0){
        output[blockIdx.x] = shmem[0];
    }
}

void reduction(double *output,double *input, int num_threads, int arr_size){
    int num_SM;
    int num_block_per_SM;
    cudaDeviceGetAttribute(&num_SM,cudaDevAttrMultiProcessorCount,0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_block_per_SM,reduction_kernel,num_threads,num_threads*sizeof(double));
    int num_blocks = min(num_block_per_SM * num_SM,(arr_size + num_threads - 1)/num_threads);

    reduction_kernel<<<num_blocks,num_threads,num_threads * sizeof(double),0>>>(output, input, arr_size);
    reduction_kernel<<<1,num_threads,num_threads * sizeof(double),0>>>(output, input, num_blocks);

}

#endif
