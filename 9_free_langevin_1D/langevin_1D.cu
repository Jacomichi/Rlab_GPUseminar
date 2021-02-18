#include <iomanip>
#include <limits>

#include "thrust_all.cuh"
#include "./utility/timer.cuh"
#include "./utility/random.cuh"

#include "./src/particle.cuh"

struct is_in{
    typedef thrust::tuple<double,double> tdouble2;
    __host__ __device__
    double operator()(tdouble2 r){
        double _x = thrust::get<0>(r);
        double _y = thrust::get<1>(r);
        return (_x * _x + _y * _y < 1.0f) ? 1.0f : 0.0f;
    }
};

int main(){
    constexpr int N = 1<<20;
    constexpr int time_steps = 10000;
    constexpr double dt = 0.01;
    constexpr double mass = 1.0;

    constexpr int threads = 1<<10;
    const int blocks = N / threads;

    Timer timer;
    cudaTimer cuda_timer;
    timer.start_record();
    cuda_timer.start_record();

    std::cout << "start" << '\n';
    Atoms atoms(N);
    atoms.fillPos(0.0);
    atoms.fillVelo(0.0);
    std::cout << "atoms create" << '\n';

    std::random_device rnd;
    setCurand<<<blocks,threads>>>(thrust::raw_pointer_cast(atoms.state),rnd(),N);
    //auto gen = createRandGenerator();

    for(int i = 0;i < time_steps;++i){
        global_updatePos<<<blocks,threads>>>(thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.vx),dt,N);
        global_updateVelo<<<blocks,threads>>>(thrust::raw_pointer_cast(atoms.vx),thrust::raw_pointer_cast(atoms.state),dt,mass,N);
        //atoms.updateVelo(gen,dt,mass);
        //atoms.updatePos(dt);
        if(i == 1000){
            thrust::copy(atoms.x,atoms.x+N, atoms.y);
        }
    }
    /*
    for(int i = 0;i < N;++i){
        std::cout << i << '\n';
        std::cout << atoms.vx[i] << '\n';
        std::cout << atoms.x[i] << '\n';
    }
    */
    //double sum = thrust::reduce(atoms.vx,atoms.vx+N,0.0,thrust::plus<double>())/static_cast<double>(N);
    double sum = thrust::inner_product(atoms.vx,atoms.vx+N, atoms.vx, 0.0)/static_cast<double>(N);
    std::cout << sum << '\n';
    thrust::transform(atoms.x,atoms.x+N,atoms.y,atoms.x,[=]__device__(auto _vx,auto _vy){return _vx - _vy;});
    double time_span = (time_steps-1000)*dt;
    double diffusion_const = thrust::inner_product(atoms.x,atoms.x+N, atoms.x, 0.0)/static_cast<double>(N)/2.0/time_span;
    std::cout << diffusion_const << '\n';
    /*
    double pi = 0.0;
    for(int i = 0; i < Tsteps;++i){
        atoms.random_uniform(gen);
        thrust::transform(atoms.first_zip_iter(),atoms.end_zip_iter(),result.begin(),is_in());
        pi += 4.0*thrust::reduce(result.begin(),result.end(),0.0,thrust::plus<double>())/N;
    }
    std::cout << std::setprecision(8) << std::scientific << pi/Tsteps << '\n';
    */

    cuda_timer.stop_record();
    timer.stop_record();

    cuda_timer.print_result();
    timer.print_result();

    //curandDestroyGenerator(gen);

    return 0;
}
