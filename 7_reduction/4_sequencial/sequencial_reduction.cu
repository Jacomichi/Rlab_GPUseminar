#include <random>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "thrust_all.cuh"
#include "reduction.cuh"

curandGenerator_t createRandGenerator(){
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    std::random_device rnd;
    curandSetPseudoRandomGeneratorSeed(gen,rnd());
    return gen;
}


int main(){
    constexpr unsigned int N = 1e7;
    constexpr int num_threads = 256;
    constexpr int trial_number = 100;

    cudaEvent_t start, stop;
    float elapse;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    thrust::device_vector<double> input(N);
    thrust::device_vector<double> output(N);

    auto gen = createRandGenerator();

    curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(&input[0]),N,0.0,1.0);
    cudaDeviceSynchronize();

    auto output_ptr = thrust::raw_pointer_cast(&output[0]);
    //trial
    for(int i = 0; i < trial_number;++i){
        thrust::copy(input.begin(),input.end(),output.begin());
        reduction(output_ptr,output_ptr, num_threads, N);
        cudaDeviceSynchronize();
    }
    std::cout << output[0]/static_cast<double>(N)  << '\n';


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapse, start, stop);
    std::cout<<"gpu :"<<elapse / static_cast<double>(trial_number) <<"ms"<<std::endl;
    float bandwidth = N * sizeof(double) / elapse / 1e6;
    std::cout <<  "bandwidth= " << bandwidth << "GB/s\n";
    // 終了処理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    curandDestroyGenerator(gen);

    return 0;
}
