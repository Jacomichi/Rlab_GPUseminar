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

    Timer timer;
    cudaTimer cuda_timer;
    timer.start_record();
    cuda_timer.start_record();

    std::cout << "start" << '\n';
    Atoms atoms(N);

    atoms.fillPos(0.0);
    atoms.fillVelo(0.0);
    std::cout << "atoms create" << '\n';

    auto gen = createRandGenerator();
    Atoms prev_pos(N);

    for(int i = 0;i < time_steps;++i){
        atoms.updateVelo(gen,dt,mass);
        atoms.updatePos(dt);

        if(i == 2000)prev_pos = atoms;

    }
    std::cout << "fin loop" << '\n';

    /*
    for(int i = 0;i < N;++i){
        std::cout << i << '\n';
        std::cout << prev_pos.vx[i] << '\n';
        std::cout << prev_pos.x[i] << '\n';
    }
    */

    double v_sq = atoms.ave_v_sq();
    std::cout << v_sq << '\n';


    double time_span = (time_steps - 2000.0)*dt;
    double diffusion_const = ave_dist(atoms,prev_pos)/time_span/6.0;
    std::cout << diffusion_const << '\n';



    cuda_timer.stop_record();
    timer.stop_record();

    cuda_timer.print_result();
    timer.print_result();

    curandDestroyGenerator(gen);

    return 0;
}
