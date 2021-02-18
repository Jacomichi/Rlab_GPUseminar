#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <algorithm>

#define NUM_LOAD 4

__inline__ __device__
double warp_reduce_sum(double sum){
    for(int offset = warpSize/2; offset > 0 ; offset >>= 1){
        int mask = __activemask();
        sum += __shfl_down_sync(mask,sum,offset);
    }
    return sum;
}

__inline__ __device__
double block_reduce_sum(double sum){
    __shared__ double shmem[32];
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;

    sum = warp_reduce_sum(sum);

    if(lane_idx == 0){
        shmem[warp_idx] = sum;
    }

    __syncthreads();

    if(warp_idx == 0){
        sum = (threadIdx.x < blockDim.x / warpSize) ? shmem[lane_idx] : 0.0;
        sum = warp_reduce_sum(sum);
    }

    return sum;
}

__device__
double grid_stride_sum(int index,int N,double *input){
    double local_sum[NUM_LOAD] = {0.0};
    for(int j = index; j < N ; j += blockDim.x * gridDim.x*NUM_LOAD){
        for(int k = 0 ; k < NUM_LOAD;++k){
            local_sum[k] += (j + k * blockDim.x * gridDim.x < N) ? input[j + k * blockDim.x * gridDim.x] : 0.0;
        }
    }

    for(int j = 1; j< NUM_LOAD; ++j){
        local_sum[0] += local_sum[j];
    }
    return local_sum[0];
}

__global__
void reduction_kernel(double *output,double *input, int N){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    double sum = 0.0;
    sum = grid_stride_sum(idx,N,input);

    sum = block_reduce_sum(sum);

    if(threadIdx.x == 0){
        output[blockIdx.x] = sum;
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
