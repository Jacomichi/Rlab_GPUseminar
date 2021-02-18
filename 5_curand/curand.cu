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
    if (rnd_x * rnd_x + rnd_y * rnd_y < 1.0){
        result[idx] = 1;
    }
}


int main(){
    constexpr unsigned int N = 1<<12;
    constexpr unsigned int num_Blocks = 1<<8;
    unsigned int threads_per_blocks = std::min(std::ceil(static_cast<double>(N)/num_Blocks),1024.0);

    //av: 平均値, ave_sq: 二乗の平均
    double av = 0.;
    double ave_sq = 0.;
    //乱数の状態

    thrust::device_vector<float> rnd_dev(N);
    thrust::device_vector<curandState> state(N);
    thrust::host_vector<float> rnd(N);
    float *rnd_dev__ptr = thrust::raw_pointer_cast(&rnd_dev[0]);
    curandState *state_ptr = thrust::raw_pointer_cast(&state[0]);

    execSetting set(num_Blocks,threads_per_blocks);

    setRand(state_ptr, set);


    //正規乱数の生成
    genrand_kernel<<<set.grid,set.block>>>(rnd_dev__ptr,state_ptr);
    thrust::copy(rnd_dev.begin(),rnd_dev.end(),rnd.begin());

    for(uint i = 0; i < N; i++){
        av += rnd[i]/(N);
        ave_sq += rnd[i] * rnd[i]/(N);
    }

    double sigma = sqrt(ave_sq - av*av);

    //平均と標準偏差を表示
    printf("av = %f\nsig = %f\n",av,sigma);
    return 0;
}
