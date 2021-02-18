#include <random>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "thrust_all.cuh"

curandGenerator_t createRandGenerator(){
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    std::random_device rnd;
    curandSetPseudoRandomGeneratorSeed(gen,rnd());
    return gen;
}

struct is_in{
    typedef thrust::tuple<float,float> tfloat2;
    __host__ __device__
    float operator()(tfloat2 r){
        float _x = thrust::get<0>(r);
        float _y = thrust::get<1>(r);
        return (_x * _x + _y * _y < 1.0f) ? 1.0f : 0.0f;
    }
};

int main(){
    constexpr unsigned int N = 1<<30;

    cudaEvent_t start, stop;
    float elapse;

    // initialize time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record initial time
    cudaEventRecord(start);


    thrust::device_vector<float> x(N);
    thrust::device_vector<float> y(N);

    auto gen = createRandGenerator();

    curandGenerateUniform(gen,thrust::raw_pointer_cast(&x[0]),N);
    curandGenerateUniform(gen,thrust::raw_pointer_cast(&y[0]),N);
    cudaDeviceSynchronize();

    /*
    auto is_in_circle = []__device__(float _x,float _y){
        return (_x * _x + _y * _y < 1.0f) ? 1.0f : 0.0f;
    };
    thrust::transform(x.begin(),x.end(),y.begin(),result.begin(),is_in_circle);
    double pi = thrust::reduce(result.begin(),result.end(),0.0f,thrust::plus<float>());
    */

    auto first = thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(x.end(),y.end()));
    double pi =thrust::transform_reduce(first,last,is_in(),0.0f,thrust::plus<float>());

    std::cout << 4.0 * pi/N << '\n';
    cudaEventRecord(stop);

    //wait until all events complete
    cudaEventSynchronize(stop);

    // calc
    cudaEventElapsedTime(&elapse, start, stop);
    std::cout<<"gpu :"<<elapse<<"ms"<<std::endl;
    // 終了処理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    curandDestroyGenerator(gen);

    return 0;
}
