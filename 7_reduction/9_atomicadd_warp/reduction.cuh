#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <algorithm>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define NUM_LOAD 4

template <typename group_t>
__inline__ __device__
double warp_reduce_sum(group_t group,double sum){
    #pragma unroll
    for(int offset = group.size()/2; offset > 0 ; offset >>= 1){
        sum += group.shfl_down(sum,offset);
    }
    return sum;
}

__device__
double grid_stride_sum(cg::thread_block block,int index,int N,double *input){
    int block_dim = block.group_dim().x;
    double local_sum[NUM_LOAD] = {0.0};
    for(int j = index; j < N ; j += block_dim * gridDim.x*NUM_LOAD){
        for(int k = 0 ; k < NUM_LOAD;++k){
            local_sum[k] += (j + k * block_dim * gridDim.x < N) ? input[j + k * block_dim * gridDim.x] : 0.0;
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
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    double sum = 0.0;
    sum = grid_stride_sum(block,idx,N,input);

    sum = warp_reduce_sum(warp,sum);

    if(warp.thread_rank() == 0){
        atomicAdd(&output[0],sum);
    }
}

void reduction(double *output,double *input, int num_threads, int arr_size){
    int num_SM;
    int num_block_per_SM;
    cudaDeviceGetAttribute(&num_SM,cudaDevAttrMultiProcessorCount,0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_block_per_SM,reduction_kernel,num_threads,num_threads*sizeof(double));
    int num_blocks = min(num_block_per_SM * num_SM,(arr_size + num_threads - 1)/num_threads);

    reduction_kernel<<<num_blocks,num_threads>>>(output, input, arr_size);
}

#endif
