#ifndef MD_HOST_CUH
#define MD_HOST_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"
#include "./particle.cuh"
#include "./list.cuh"
#include "./setting.cuh"

void h_create_list_full_search(Atoms &atoms,List &list,Setting &setting){
  list.refresh(atoms.N);
  create_list<<<setting.numBlocks,setting.numThreads>>>(
    thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
  thrust::raw_pointer_cast(list.list),setting.N,setting.L,list.list_size,list.cutoff);
}

void h_update_verlet_list(Atoms &atoms,List &list,Mesh &mesh, Setting &setting){
  list.refresh(atoms.N);
  mesh.refresh();

  mesh.check_index(atoms);
  mesh.check_num_atoms_in_mesh();

  updateVerletList<<<setting.numBlocks,setting.numThreads>>>(
    thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(mesh.mesh_begin),thrust::raw_pointer_cast(mesh.mesh_end),
    thrust::raw_pointer_cast(mesh.atoms_id),thrust::raw_pointer_cast(list.list),setting.L,
     list.cutoff,list.list_size,mesh.num_mesh_per_axis,mesh.mesh_length
   );
}

void EoM_MD(Atoms &atoms,List &list,Mesh &mesh, Setting &setting){
  atoms.updateVeloMD(setting.dt);
  
  calc_force_with_list<<<setting.numBlocks,setting.numThreads>>>(
    thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(atoms.fx),thrust::raw_pointer_cast(atoms.fy),
    thrust::raw_pointer_cast(atoms.sigma),setting.L,
    thrust::raw_pointer_cast(list.list),list.list_size
  );

  atoms.updateVeloMD(setting.dt);
  atoms.updatePosMD(setting.dt);
  atoms.set_in_box(setting);
}

void EoM_BD(Atoms &atoms,List &list,Mesh &mesh, Setting &setting,curandGenerator_t &gen){
  calc_force_with_list<<<setting.numBlocks,setting.numThreads>>>(
    thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(atoms.fx),thrust::raw_pointer_cast(atoms.fy),
    thrust::raw_pointer_cast(atoms.sigma),setting.L,
    thrust::raw_pointer_cast(list.list),list.list_size
  );

  atoms.updateVeloBD(gen,setting.dt);
  atoms.updatePosBD(setting.dt);
  atoms.set_in_box(setting);
}


void quench(Atoms &atoms,List &list,Mesh &mesh, Setting &setting){
  calc_force_Harmonic<<<setting.numBlocks,setting.numThreads>>>(
    thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(atoms.fx),thrust::raw_pointer_cast(atoms.fy),
    thrust::raw_pointer_cast(atoms.sigma),setting.L,
    thrust::raw_pointer_cast(list.list),list.list_size
  );

  atoms.updatePosQuench(setting.dt);
  atoms.set_in_box(setting);
}

double h_calc_potential_energy(Atoms &atoms,Setting &setting,List &list){
  calc_potential_with_list<<<setting.numBlocks,setting.numThreads>>>(
    thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(atoms.potential),thrust::raw_pointer_cast(atoms.sigma),
    setting.L,thrust::raw_pointer_cast(list.list),list.list_size);
  double ave_pot =  atoms.average_potential();
  return ave_pot;
}

double h_calc_potential_energy_harmonic(Atoms &atoms,Setting &setting,List &list){
  calc_potential_Harmonic<<<setting.numBlocks,setting.numThreads>>>(
    thrust::raw_pointer_cast(atoms.x),thrust::raw_pointer_cast(atoms.y),
    thrust::raw_pointer_cast(atoms.potential),thrust::raw_pointer_cast(atoms.sigma),
    setting.L,thrust::raw_pointer_cast(list.list),list.list_size);
  double ave_pot =  atoms.average_potential();
  return ave_pot;
}

#endif
