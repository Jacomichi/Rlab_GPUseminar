#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <random>
#include <iostream>
#include <algorithm>

#include <curand.h>
#include <curand_kernel.h>

#include "thrust_all.cuh"

//curandStateの初期化
__global__ void setCurand(unsigned long long seed, curandState *state){
    uint i_global = threadIdx.x + blockIdx.x*blockDim.x;
    curand_init(seed, i_global, 0, &state[i_global]);
}

struct execSetting{
    dim3 grid;
    dim3 block;
    execSetting(dim3 _grid,dim3 _block){
        grid = _grid;
        block = block;
    }
    execSetting(int gridsize,int blocksize){
        dim3 _grid(gridsize);
        grid = _grid;
        dim3 _block(blocksize);
        block = _block;
    }
};

void setRand(curandState *state,execSetting set){
    std::random_device _rnd;
    setCurand<<<set.grid,set.block>>>(_rnd(), state);
}

template <typename T>
std::size_t check_size(int N){
    return N * sizeof(T);
}

//一様乱数を返す
__global__ void genrand_kernel(float *result, curandState *state){

    uint i_global = threadIdx.x + blockIdx.x*blockDim.x;
    result[i_global] = curand_normal(&state[i_global]);
}

__global__ void calc_pi(float *result, curandState *state){
    auto idx = threadIdx.x + blockIdx.x*blockDim.x;
    auto rnd_x = curand_uniform(&state[idx]);
    auto rnd_y = curand_uniform(&state[idx]);
    result[idx] = (rnd_x * rnd_x + rnd_y * rnd_y < 1.0f) ? 1.0f : 0.0f;
}


int main(){
    constexpr unsigned int N = 1<<24;
    constexpr unsigned int num_Blocks = 1<<14;
    unsigned int threads_per_blocks = std::min(std::ceil(static_cast<double>(N)/num_Blocks),1024.0);

    cudaEvent_t start, stop;
    float elapse;

    // initialize time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record initial time
    cudaEventRecord(start);

    thrust::device_vector<float> result(N);
    thrust::device_vector<curandState> state(N);
    thrust::host_vector<float> hresult(N);
    float *result_ptr = thrust::raw_pointer_cast(&result[0]);
    curandState *state_ptr = thrust::raw_pointer_cast(&state[0]);

    execSetting set(num_Blocks,threads_per_blocks);

    setRand(state_ptr, set);

    calc_pi<<<set.grid,set.block>>>(result_ptr,state_ptr);
    double pi = thrust::reduce(result.begin(),result.end(),0.0f,thrust::plus<float>());

    std::cout << 4.0/N*pi << '\n';
    cudaEventRecord(stop);

    //wait until all events complete
    cudaEventSynchronize(stop);

    // calc
    cudaEventElapsedTime(&elapse, start, stop);
    std::cout<<"gpu :"<<elapse<<"ms"<<std::endl;
    // 終了処理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
