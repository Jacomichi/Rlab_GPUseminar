#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"

struct size_t2{
    thrust::device_ptr<double> x;
    thrust::device_ptr<double> y;
    size_t2(thrust::device_ptr<double> _x,thrust::device_ptr<double> _y){
        x =_x;
        y =_y;
    }
};

struct Atoms{
    unsigned int N;
    thrust::device_ptr<double> x;
    thrust::device_ptr<double> vx;
    thrust::device_ptr<double> y;
    thrust::device_ptr<double> rnd;
    thrust::device_ptr<curandState> state;

    Atoms(size_t _N){
        N = _N;
        x = thrust::device_malloc<double>(_N);
        vx = thrust::device_malloc<double>(_N);
        y = thrust::device_malloc<double>(_N);
        rnd = thrust::device_malloc<double>(_N);
        state = thrust::device_malloc<curandState>(_N);
    }

    ~Atoms(){
        thrust::device_free(x);
        thrust::device_free(y);
        thrust::device_free(vx);
        thrust::device_free(rnd);
    }

    void random_uniform(curandGenerator_t gen){
        curandGenerateUniformDouble(gen,thrust::raw_pointer_cast(x),N);
        curandGenerateUniformDouble(gen,thrust::raw_pointer_cast(y),N);
    }

    void gauss(curandGenerator_t gen){
        curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(rnd),N,0.0,1.0);
    }

    size_t2 end(){
        return size_t2(x+N,y+N);
    }

    size_t2 first(){
        return size_t2(x,y);
    }

    auto first_zip_iter(){
        return thrust::make_zip_iterator(thrust::make_tuple(x,y));
    }

    auto end_zip_iter(){
        return thrust::make_zip_iterator(thrust::make_tuple(x+N,y+N));
    }

    void updatePos(double dt){
        thrust::transform(x,x+N,vx,x,[=]__device__(auto _x,auto _vx){return _x + _vx*dt;});
    }

    void updateVelo(curandGenerator_t gen,double dt,double mass){
        double noise_strength = sqrt(2.0 * dt);
        curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(rnd),N,0.0,noise_strength);
        thrust::transform(vx,vx+N,rnd,vx,[=]__device__(auto _vx,auto _rnd){
            return _vx * (1.0 - dt/mass) + _rnd/mass;});
    }


    void fillPos(double i){
        thrust::fill(x,x+N,i);
        thrust::fill(y,y+N,i);
    }

    void fillVelo(double i){
        thrust::fill(vx,vx+N,i);
    }

};



__global__ void global_updateVelo(double *vx,curandState *state,double dt,double mass,int N){
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        vx[idx] = vx[idx] * (1.0 - dt/mass) + sqrt(2.0 * dt)*curand_normal_double(&state[idx])/mass;
    }
}

__global__
void setCurand(curandState *state,unsigned long long seed,int N){
    auto idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < N){
            curand_init(seed, idx, 0, &state[idx]);
        }
}

__global__ void global_updatePos(double *x,double *vx,double dt,int N){
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N){
        x[idx] = x[idx] + vx[idx] * dt;
    }
}


#endif
