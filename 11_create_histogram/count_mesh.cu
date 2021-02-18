#include <iomanip>
#include <limits>

#include "thrust_all.cuh"
#include "./utility/timer.cuh"
#include "./utility/random.cuh"

#include "./src/particle.cuh"
#include "./src/mesh.cuh"

int main(){
    constexpr int N = 1<<15;
    constexpr int mesh_num = 10;
    constexpr double L = 1.0;

    Timer timer;
    cudaTimer cuda_timer;


    std::cout << "start" << '\n';
    Atoms atoms(N);
    atoms.fillVelo(0.0);

    std::cout << "atoms create" << '\n';

    auto gen = createRandGenerator();

    Mesh mesh(N,mesh_num,L);
    timer.start_record();
    cuda_timer.start_record();
    for(int i =0;i < 1000;++i){
        atoms.random_uniform(gen);
        mesh.check_index(atoms);
        mesh.check_num_atoms_in_mesh();
    }
    cuda_timer.stop_record();
    timer.stop_record();

    cuda_timer.print_result();
    timer.print_result();

    /*
    int mesh_id = 1;
    std::cout << "check mesh" << '\n';
    for(int p_idx = mesh.mesh_begin[mesh_id];p_idx != mesh.mesh_end[mesh_id];++p_idx){
        int i = p_idx;
        int particle_id = mesh.atoms_id[i];
        std::cout << "mesh ID : " << mesh.mesh_index_of_atom[i] << " (x,y) = (" << atoms.x[particle_id] <<" , " << atoms.y[particle_id] << ") particle ID : "<< mesh.atoms_id[i] << '\n';
    }
    */

    /*
    std::cout << "make histogram" << '\n';
    thrust::device_vector<int> histogram(mesh_num * mesh_num);
    create_histogram(mesh,histogram);
    for(int i = 0;i<histogram.size();++i){
        std::cout << "mesh ID : " << i << " num of particle in mesh : " << histogram[i] << '\n';
    }
    double ave = thrust::reduce(histogram.begin(),histogram.end())/static_cast<double>(histogram.size());
    //ラムダ式の引数をdoubleにしとかないとオーバーフローする。
    double var = thrust::transform_reduce(histogram.begin(),histogram.end(),[=]__device__(double x){return x*x;},0.0,thrust::plus<double>())/static_cast<double>(histogram.size());
    std::cout << "ave from data " << ave << '\n';
    std::cout << "variance from data " << var - ave*ave << '\n';
    std::cout << "ave from theory " << static_cast<double>(N)/(mesh_num * mesh_num) << '\n';
    */


    curandDestroyGenerator(gen);

    return 0;
}
