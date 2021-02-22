#include <iomanip>
#include <limits>
#include <algorithm>

#include "thrust_all.cuh"
#include "./utility/timer.cuh"
#include "./utility/random.cuh"

#include "./src/particle.cuh"
#include "./src/mesh.cuh"
#include "./src/list.cuh"

int main(){
    constexpr int N = 1<<20;
    //constexpr int mesh_num = 10;
    constexpr double rho = 0.8;
    double L = sqrt(N/rho);
    double mesh_size = 2.5;
    int mesh_num = floor(L/mesh_size);
    constexpr int numThreads = 1<<10;
    constexpr int list_size = 30;

    cudaTimer timer;
    cudaTimer timer2;
    cudaTimer timer3;

    std::cout << "start" << '\n';
    Atoms atoms(N);
    atoms.fillVelo(0.0);

    Mesh mesh(N,mesh_num,L);

    std::cout << "atoms create" << '\n';

    auto gen = createRandGenerator();
    atoms.random_uniform(gen,L);

    timer.start_record();

    //create list O(N^2)
    List list(N,list_size,1.0);
    create_list<<<(N + numThreads - 1)/numThreads,numThreads>>>(thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(list.list),N,L,list.list_size,list.cutoff);

    timer.stop_record();
    timer.print_result();


    /*
    for(int k = 0;k < N * cell_size; ++k){
        std::cout << list.list[k] << '\n';
    }
    */


    timer2.start_record();
    //create list O(N)
    List mesh_list(N,list_size,1.0);
    mesh.check_index(atoms);
    mesh.check_num_atoms_in_mesh();
    timer2.stop_record();
    timer2.print_result();

    timer3.start_record();
    updateVerletList<<<(N + numThreads - 1)/numThreads,numThreads>>>(thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(mesh.mesh_begin),thrust::raw_pointer_cast(mesh.mesh_end),thrust::raw_pointer_cast(mesh.atoms_id), thrust::raw_pointer_cast(mesh_list.list),
    L,mesh_list.cutoff,mesh_list.list_size,mesh.num_mesh_per_axis,mesh.mesh_length);

    timer3.stop_record();
    timer3.print_result();

    /*
    std::vector<int> full_search;
    std::vector<int> grid_search;
    for(int i =0;i < N;++i){
        //std::cout << i << " : (" << atoms.x[i] << "," << atoms.y[i] << ")" << '\n';
        int pair_list_size = mesh_list.list[i * list_size];
        int pair_list_size_full_search = list.list[i * list_size];

        if(!(pair_list_size == pair_list_size_full_search)){
            std::cout << i <<" : size is defferent" << '\n';
            for(int j = 1;j < pair_list_size_full_search ; ++j){
                int pair_id = mesh_list.list[i * list_size + j];
                std::cout << "----using grid : " << pair_id ;
                pair_id = list.list[i * list_size + j];
                std::cout << " full search : " << pair_id << '\n';
            }
        }

        /*
        for(int j = 1;j < pair_list_size_full_search ; ++j){
            int pair_id = mesh_list.list[i * list_size + j];
            full_search.push_back(pair_id);
            std::cout << "----using grid : " << pair_id ;
            pair_id = list.list[i * list_size + j];
            grid_search.push_back(pair_id);
            std::cout << " full search : " << pair_id << '\n';
        }
        */
        /*
        std::sort(full_search.begin(),full_search.end());
        std::sort(grid_search.begin(),grid_search.end());
        if(!(full_search == grid_search)){
            std::cout << "Creating verlet list is something wrong" << '\n';
        }
        full_search.clear();
        grid_search.clear();
    }*/



    /*
    for(int k = 0;k < N * cell_size; ++k){
        std::cout << mesh_list.list[k] << '\n';
    }

    int mesh_id = 1;
    std::cout << "check mesh" << '\n';
    for(int p_idx = mesh.mesh_begin[mesh_id];p_idx != mesh.mesh_end[mesh_id];++p_idx){
        int i = p_idx;
        int particle_id = mesh.atoms_id[i];
        std::cout << "mesh ID : " << mesh.mesh_index_of_atom[i] << " (x,y) = (" << atoms.x[particle_id] <<" , " << atoms.y[particle_id] << ") particle ID : "<< mesh.atoms_id[i] << '\n';
    }
    */



    curandDestroyGenerator(gen);

    return 0;
}
