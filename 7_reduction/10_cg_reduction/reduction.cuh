#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define NUM_LOAD 4


__inline__ __device__
double block_reduce_sum(cg::thread_block block,double sum){
    __shared__ double shmem[32];
    int warp_idx = block.thread_index().x / warpSize;

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    //sum = warp_reduce_sum(warp,sum);
    sum = cg::reduce(warp,sum,cg::plus<double>());

    if(warp.thread_rank() == 0){
        shmem[warp_idx] = sum;
    }

    block.sync();

    if(warp_idx == 0){
        sum = (threadIdx.x < block.group_dim().x / warpSize) ? shmem[warp.thread_rank()] : 0.0;
        sum = cg::reduce(warp,sum,cg::plus<double>());
    }

    return sum;
}

__device__
double grid_stride_sum(cg::thread_block block,int index,int N,double *input){
    double local_sum[NUM_LOAD] = {0.0};
    for(int j = index; j < N ; j += block.group_dim().x * gridDim.x*NUM_LOAD){
        for(int k = 0 ; k < NUM_LOAD;++k){
            local_sum[k] += (j + k * block.group_dim().x * gridDim.x < N) ? input[j + k * block.group_dim().x * gridDim.x] : 0.0;
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
    cg::thread_block block = cg::this_thread_block();

    double sum = 0.0;
    sum = grid_stride_sum(block,idx,N,input);

    sum = block_reduce_sum(block,sum);

    if(block.thread_index().x == 0){
        output[block.group_index().x] = sum;
    }
}

void reduction(double *output,double *input, int num_threads, int arr_size){
    int num_SM;
    int num_block_per_SM;
    cudaDeviceGetAttribute(&num_SM,cudaDevAttrMultiProcessorCount,0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_block_per_SM,reduction_kernel,num_threads,num_threads*sizeof(double));
    int num_blocks = min(num_block_per_SM * num_SM,(arr_size + num_threads - 1)/num_threads);

    reduction_kernel<<<num_blocks,num_threads>>>(output, input, arr_size);
    reduction_kernel<<<1,num_threads>>>(output, input, num_blocks);
}

#endif
