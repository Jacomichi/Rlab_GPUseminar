#ifndef GRID_CUH
#define GRID_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"
#include "./particle.cuh"

struct Grid{
    int grid_num;
    //Here we consider only square meshes
    int maxParticlesPerCell;
    double grid_length;
    thrust::device_ptr<int> grid_counter;
    thrust::device_ptr<int> grid_list;

    Grid(int _grid_num,double L,int _maxParticlesPerCell){
        grid_num = _grid_num;
        maxParticlesPerCell = _maxParticlesPerCell;
        grid_length = L / static_cast<double>(_grid_num);
        grid_counter = thrust::device_malloc<int>(grid_num * grid_num);
        grid_list = thrust::device_malloc<int>(grid_num * grid_num);
    }

    ~Grid(){
        thrust::device_free(grid_counter);
        thrust::device_free(grid_list);
    }
};


#endif
