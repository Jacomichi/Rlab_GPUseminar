#ifndef LIST_CUH
#define LIST_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"
#include "./particle.cuh"
#include "structs.h"
#include <cassert>

struct List{
  int list_size;
	double cutoff;
	uint refresh_span;
  thrust::device_ptr<int> list;

  List(int N, int _list_size,double _cutoff,uint _refresh_span ){
    list_size = _list_size;
		refresh_span = _refresh_span;
		cutoff = _cutoff;
    list = thrust::device_malloc<int>(N * _list_size);
		thrust::fill(list,list + N * _list_size,-1);
  }

  ~List(){
        thrust::device_free(list);
    }

		void refresh(int N){
			thrust::fill(list,list + N * list_size,-1);
		}

};

//create list from full search which is O(N^2)
__global__
void create_list(double *x,double *y,int *list,int N,double L,int list_size,double list_cutoff){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
	double2 pos1 = make_double2(x[index],y[index]);
	int counter = 1;
	for(int index2 = 0;index2 < N;++index2){
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
						double2 relative = pos1 - pos2;
						relative.x = relative.x - L*floor(relative.x/L+0.5);
						relative.y = relative.y - L*floor(relative.y/L+0.5);
						double r = sqrt(dot(relative, relative));
						if(r < list_cutoff) {
							list[index*list_size + counter] = index2;
							counter += 1;
						}
        }
  	}
		assert(counter < list_size);
		list[index*list_size] = counter;

}


// create list from grid search which is O(N)
__device__
int2 calcGridPos(double2 p,double grid_length) {
	int2 gridPos;
	gridPos.x = floor(p.x / grid_length);
	gridPos.y = floor(p.y / grid_length);
	return gridPos;
}

__device__
uint calcGridAddress(int2 gridPos,int grid_num_per_axis) {
	return gridPos.y * grid_num_per_axis + gridPos.x;
}

__device__
int searchCell(int2 gridPos, int index, double2 pos1, double *x,double *y,int *list,double L,double list_cutoff,int list_size,int counter,int *mesh_begin,int *mesh_end,int *atoms_id,int grid_num_per_axis) {
	if(gridPos.x == -1) gridPos.x = grid_num_per_axis-1;
	if(gridPos.x == grid_num_per_axis) gridPos.x = 0;
	if(gridPos.y == -1) gridPos.y = grid_num_per_axis-1;
	if(gridPos.y == grid_num_per_axis) gridPos.y = 0;

	int mesh_id = calcGridAddress(gridPos, grid_num_per_axis);
	int local_counter = 0;
	for(int i=mesh_begin[mesh_id]; i !=mesh_end[mesh_id]; ++i) {
		int index2 = atoms_id[i];
		if(index2 != index) {
			double2 pos2 = make_double2(x[index2],y[index2]);
			double2 relative = pos2 - pos1;
			relative.x = relative.x - L*floor(relative.x/L+0.5);
			relative.y = relative.y - L*floor(relative.y/L+0.5);
			double r = sqrt(dot(relative, relative));
			if(r < list_cutoff) {
				list[index*list_size + counter + local_counter] = index2;
				local_counter += 1;
			}
		}
	}
	//printf("%d\n",counter );
	return local_counter;
}

__global__
void updateVerletList(double *x,double *y, int *mesh_begin,int *mesh_end,int *atoms_id, int *list,double L,double list_cutoff,int list_size,int grid_num_per_axis,double grid_length) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	double2 pos1 = make_double2(x[index],y[index]);
	int2 gridPos = calcGridPos(pos1,grid_length);
	int counter = 1;
	for(int grid_y=-1; grid_y<=1; ++grid_y){
		for(int grid_x=-1; grid_x<=1; ++grid_x){
			counter += searchCell(gridPos + make_int2(grid_x,grid_y), index, pos1, x,y, list,L,list_cutoff,list_size,counter,mesh_begin,mesh_end,atoms_id,grid_num_per_axis);
		}
	}
	list[index*list_size] = counter;
}

#endif
