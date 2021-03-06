#ifndef MESH_CUH
#define MESH_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"
#include "./particle.cuh"


struct Search_atoms_index{
    unsigned int num_mesh;
    double mesh_size;

    Search_atoms_index(uint _num_mesh,double _mesh_size){
        num_mesh = _num_mesh;
        mesh_size =_mesh_size;
    }
    typedef thrust::tuple<double,double> tuple_double2;
    __host__ __device__
    unsigned int operator()(const tuple_double2& r){
        double x = thrust::get<0>(r);
        double y = thrust::get<1>(r);
        int mesh_x = x/mesh_size;
        int mesh_y = y/mesh_size;
        return mesh_y * num_mesh + mesh_x;
    }
};

struct Mesh{
    unsigned int num_atoms;
    //Here we consider only square meshes
    unsigned int num_mesh;
    double mesh_size;
    thrust::device_ptr<int> mesh_index_of_atom;
    thrust::device_ptr<int> atoms_id;
    thrust::device_ptr<int> mesh_begin;
    thrust::device_ptr<int> mesh_end;

    Mesh(int _N,int _Num_mesh,double L){
        num_atoms = _N;
        num_mesh = _Num_mesh;
        mesh_size = L / static_cast<double>(_Num_mesh);
        mesh_index_of_atom = thrust::device_malloc<int>(_N);
        atoms_id = thrust::device_malloc<int>(_N);
        thrust::sequence(atoms_id,atoms_id+num_atoms);
        mesh_begin = thrust::device_malloc<int>(_Num_mesh*_Num_mesh);
        mesh_end = thrust::device_malloc<int>(_Num_mesh*_Num_mesh);
    }

    ~Mesh(){
        thrust::device_free(mesh_index_of_atom);
        thrust::device_free(atoms_id);
        thrust::device_free(mesh_begin);
        thrust::device_free(mesh_end);
    }

    void check_index(const Atoms &atoms){
        int N = atoms.N;

        auto first = thrust::make_zip_iterator(thrust::make_tuple(atoms.x,atoms.y));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(atoms.x + N,atoms.y + N));
        auto search = Search_atoms_index(num_mesh,mesh_size);
        thrust::transform(first,last,mesh_index_of_atom,search);
    }

    void check_num_atoms_in_mesh(){
        auto atoms_mesh_first = mesh_index_of_atom;
        auto atoms_mesh_end = mesh_index_of_atom+num_atoms;

        //メッシュの番号を使って、粒子IDをソート。
        //atoms_idの中身が粒子のindexに対応。
        //atoms_id自体のindexは、メッシュ0から入っている粒子数をカウントしたもの。
        //mesh_index_of_atom : {4,0,2,1,1,3}
        //atoms_id           : {0,1,2,3,4,5}
        //->mesh_index_of_atom : {0,1,1,2,3,4}
        //  atoms_id           : {1,3,4,2,5,0}
        thrust::sort_by_key(atoms_mesh_first,atoms_mesh_end,atoms_id);

        //全てのセルにつて上限と下限を調べる。
        //上限もしくは、下限だけだと中身がないと気にバグる。
        thrust::counting_iterator<unsigned int> search_begin(0);
        thrust::lower_bound(atoms_mesh_first,atoms_mesh_end,search_begin,search_begin + num_mesh*num_mesh,mesh_begin);
        thrust::upper_bound(atoms_mesh_first,atoms_mesh_end,search_begin,search_begin + num_mesh*num_mesh,mesh_end);
    }
};


void create_histogram(const Mesh &mesh,thrust::device_vector<int> &histogram){
    transform(mesh.mesh_end,mesh.mesh_end + mesh.num_mesh * mesh.num_mesh,mesh.mesh_begin,histogram.begin(),thrust::minus<int>());
}


#endif
