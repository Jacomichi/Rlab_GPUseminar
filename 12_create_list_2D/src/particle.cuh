#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include "./../thrust_all.cuh"
#include "./../utility/random.cuh"
#include "./box.cuh"
#include "./structs.h"

constexpr double WCA_cutoff =1.122462048309373; //pow(2,1/6)

struct size_t2{
    thrust::device_ptr<double> x;
    thrust::device_ptr<double> y;
    size_t2(thrust::device_ptr<double> _x,thrust::device_ptr<double> _y){
        x =_x;
        y =_y;
    }
};


void d_copy(thrust::device_ptr<double> src,thrust::device_ptr<double> dst,int N){
    thrust::copy(src,src + N,dst);
}

struct Atoms{
    unsigned int N;
    thrust::device_ptr<double> x;
    thrust::device_ptr<double> y;
    thrust::device_ptr<double> vx;
    thrust::device_ptr<double> vy;
    thrust::device_ptr<double> random_fx;
    thrust::device_ptr<double> random_fy;
    thrust::device_ptr<double> fx;
    thrust::device_ptr<double> fy;
    thrust::device_ptr<double> potential;
    thrust::device_ptr<double> sigma;

    Atoms(size_t _N){
        N = _N;
        x = thrust::device_malloc<double>(_N);
        y = thrust::device_malloc<double>(_N);
        vx = thrust::device_malloc<double>(_N);
        vy = thrust::device_malloc<double>(_N);
        random_fx = thrust::device_malloc<double>(_N);
        random_fy = thrust::device_malloc<double>(_N);
        fx = thrust::device_malloc<double>(_N);
        fy = thrust::device_malloc<double>(_N);
        potential = thrust::device_malloc<double>(_N);
        sigma = thrust::device_malloc<double>(_N);
    }

    //コピーコンストラクタ
    Atoms(const Atoms &_atoms){
        N = _atoms.N;
        x = thrust::device_malloc<double>(N);
        y = thrust::device_malloc<double>(N);
        vx = thrust::device_malloc<double>(N);
        vy = thrust::device_malloc<double>(N);
        random_fx = thrust::device_malloc<double>(N);
        random_fy = thrust::device_malloc<double>(N);
        fx = thrust::device_malloc<double>(N);
        fy = thrust::device_malloc<double>(N);
        potential = thrust::device_malloc<double>(N);
        sigma = thrust::device_malloc<double>(N);
        d_copy(_atoms.x,x,N);
        d_copy(_atoms.y,y,N);
        d_copy(_atoms.vx,vx,N);
        d_copy(_atoms.vy,vy,N);
        d_copy(_atoms.random_fx,random_fx,N);
        d_copy(_atoms.random_fy,random_fy,N);
        d_copy(_atoms.fx,fx,N);
        d_copy(_atoms.fy,fy,N);
        d_copy(_atoms.potential,potential,N);
        d_copy(_atoms.sigma,sigma,N);
    };

    Atoms &operator=(const Atoms &_atoms) {
        if(this != &_atoms) {
            this->N = _atoms.N;
            d_copy(_atoms.x,this->x,N);
            d_copy(_atoms.y,this->y,N);
            d_copy(_atoms.vx,this->vx,N);
            d_copy(_atoms.vy,this->vy,N);
            d_copy(_atoms.random_fx,this->random_fx,N);
            d_copy(_atoms.random_fy,this->random_fy,N);
            d_copy(_atoms.fx,this->fx,N);
            d_copy(_atoms.fy,this->fy,N);
            d_copy(_atoms.potential,this->potential,N);
            d_copy(_atoms.sigma,this->sigma,N);
        }
        return *this;
    }

    ~Atoms(){
        thrust::device_free(x);
        thrust::device_free(y);
        thrust::device_free(vx);
        thrust::device_free(vy);
        thrust::device_free(random_fx);
        thrust::device_free(random_fy);
        thrust::device_free(fx);
        thrust::device_free(fy);
        thrust::device_free(potential);
        thrust::device_free(sigma);
    }

    void random_uniform(curandGenerator_t gen,double L){
        curandGenerateUniformDouble(gen,thrust::raw_pointer_cast(x),N);
        curandGenerateUniformDouble(gen,thrust::raw_pointer_cast(y),N);
        auto adapt_system_size = [=]__device__(double pos){return pos*L;};
        thrust::transform(x,x+N,x,adapt_system_size);
        thrust::transform(y,y+N,y,adapt_system_size);
    }


    void create_square_lattice(Box box){
        double L = box.L;
        int num_atoms_per_axis = ceil(sqrt(N));
        //端の粒子が重ならないように、系を少し小さく見積もる。
        double lattice_size = (L - 1.0) / static_cast<double>(num_atoms_per_axis);
        std::cout << "lattice size " << lattice_size<< '\n';
        thrust::counting_iterator<int> id_iter(0);
        auto y_pos = [=]__device__(int atom_id){return static_cast<double>((atom_id /num_atoms_per_axis)) * lattice_size;};
        auto x_pos = [=]__device__(int atom_id){
            return static_cast<double>((atom_id %num_atoms_per_axis)) * lattice_size;
        };
        thrust::transform(id_iter,id_iter+N,x,x_pos);
        thrust::transform(id_iter,id_iter+N,y,y_pos);
    }

    void set_in_box(Box box){
        double L = box.L;
      auto PBC_pos = [=]__device__(double r){
        return r - floor(r/L)*L;
      };

      thrust::transform(x,x+N,x,PBC_pos);
      thrust::transform(y,y+N,y,PBC_pos);
    }

    void create_thermal_noise(curandGenerator_t gen,double noise_strength){
        curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(random_fx),N,0.0,noise_strength);
        curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(random_fy),N,0.0,noise_strength);
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
        //TO DO : using async transform makes this function much faster probably
        auto update_position = [=]__device__(auto _r,auto _v){return _r + _v*dt;};
        thrust::transform(x,x+N,vx,x,update_position);
        thrust::transform(y,y+N,vy,y,update_position);
    }

    void updateVelo(curandGenerator_t gen,double dt,double mass){
        double noise_strength = sqrt(2.0 * dt);
        create_thermal_noise(gen,noise_strength);
        auto update_velocity = [=]__device__(auto _v,auto _rnd){
            return _v * (1.0 - dt/mass) + _rnd/mass;};
        thrust::transform(vx,vx+N,random_fx,vx,update_velocity);
        thrust::transform(vy,vy+N,random_fy,vy,update_velocity);
    }

    void monodisperse(){
        thrust::fill(sigma,sigma+N,1.0);
    }

    void binary(){
        int half = int(N/2) + 1;
        thrust::fill(sigma,sigma+half,1.0);
        thrust::fill(sigma+half,sigma + N,1.4);
    }

    void fillPos(double val){
        thrust::fill(x,x+N,val);
        thrust::fill(y,y+N,val);
    }

    void fillVelo(double val){
        thrust::fill(vx,vx+N,val);
        thrust::fill(vy,vy+N,val);
    }

    double average_kinetic(){
        double vx_sq = thrust::inner_product(vx, vx + N, vx, 0.0);
        double vy_sq = thrust::inner_product(vy, vy + N, vy, 0.0);

        return (vx_sq + vy_sq )/(2.0 * static_cast<double>(N));
    }

    double average_potential(){
        return thrust::reduce(potential,potential + N)/(static_cast<double>(N));
    }
};



struct distance{
    typedef thrust::tuple<double,double,double,double> tuple_double4;
    __host__ __device__
    double operator()(const tuple_double4 &r){
        double _x1 = thrust::get<0>(r);
        double _y1 = thrust::get<1>(r);
        double _x2 = thrust::get<2>(r);
        double _y2 = thrust::get<3>(r);
        double disp = sqrt((_x1 - _x2) * (_x1 - _x2) + (_y1 - _y2)*(_y1 - _y2));
        return disp;
    }
};

double ave_dist(const Atoms &atoms1,const Atoms &atoms2){
    size_t N = atoms1.N;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(atoms1.x,atoms1.y ,atoms2.x,atoms2.y));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(atoms1.x + N,atoms1.y + N ,atoms2.x + N ,atoms2.y + N ));

    double disp = thrust::transform_reduce(first,last,distance(),0.0,thrust::plus<double>());
    return disp/(static_cast<double>(N));
}

struct _dist2D{
    typedef thrust::tuple<double,double> tuple_double2;
    __host__ __device__
    double operator()(const tuple_double2& r){
        double x = thrust::get<0>(r);
        double y = thrust::get<1>(r);
        return sqrt(x * x + y * y );
    }
};

double dist2D(const Atoms &atoms){
    int N = atoms.N;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(atoms.x,atoms.y));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(atoms.x + N,atoms.y + N));
    double disp = thrust::transform_reduce(first,last,_dist2D(),0.0,thrust::plus<double>());

    return disp/(static_cast<double>(N));
}

__device__
double2 calcForceWCA(double2 pos1, double2 pos2, double size,double L,double cutoff) {
	double2 force = make_double2(0.0);
	double2 dr = pos1 - pos2;
	dr.x = dr.x - L*floor(dr.x/L+0.5);
	dr.y = dr.y - L*floor(dr.y/L+0.5);
	double r = sqrt(dot(dr, dr));
	if(r < size*cutoff) {
		dr = dr / r;
        double invr = size/r;
        invr *= invr;
		double invR = invr * invr * invr;
		force = (24.0*invR*(2.0*invR - 1.0)/r)*dr;
	}
	return force;
}

__global__
void calc_force(double *x,double *y, double *fx, double *fy,double *sigma,int N,double L, double cutoff) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    double2 force = make_double2(0.0);

    for(int index2 = 0;index2 < N;++index2){
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
            double size = (size1 + sigma[index2])/2.0;
            force += calcForceWCA(pos1, pos2, size,L,cutoff);
        }
    }
    fx[index] = force.x;
    fy[index] = force.y;
}




//Calc force with mesh
/*
__device__
int2 calcGridPos(double2 p,double cell_size) {
	int2 gridPos;
	gridPos.x = floor(p.x / cell_size);
	gridPos.y = floor(p.y / cell_size);
	return gridPos;
}

__device__
uint calcGridAddress(int2 gridPos,int grid_num) {
	return gridPos.y * grid_num + gridPos.x;
}


__device__
double2 calcForceWCA_with_mesh(int2 gridPos, int index, double size1, double2 pos1, double *x,double *y, int *mesh_begin,int *mesh_end,int *atoms_id,double L,double cutoff,int grid_num) {
    double2 force = make_double2(0.0);
    if(gridPos.x == -1) gridPos.x = grid_num-1;
	if(gridPos.x == grid_num) gridPos.x = 0;
	if(gridPos.y == -1) gridPos.y = grid_num-1;
	if(gridPos.y == grid_num) gridPos.y = 0;

	uint mesh_id = calcGridAddress(gridPos, grid_num);

	for(uint p_idx=mesh_begin[mesh_id]; p_idx !=mesh_end[mesh_id]; ++p_idx) {
		uint index2 = atoms_id[p_idx];
		if(index2 != index) {
			double2 pos2 = make_double2(x[index2],y[index2]);
			double size2 = 1.0;
			double s = (size1 + size2) / 2.0;
			force += calcForceWCA(pos1, pos2, s,L,cutoff);
		}
	}
	return force;
}

__global__
void calc_force_with_mesh(double *x,double *y, double *fx, double *fy,double *sigma,double L, double cutoff,int *mesh_begin,int *mesh_end,int *atoms_id,int grid_num) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    int2 gridPos = calcGridPos(pos1,grid_num);
    double2 force = make_double2(0.0);

    for(int grid_y=-1; grid_y<=1; ++grid_y){
		for(int grid_x=-1; grid_x<=1; ++grid_x){
			force += calcForceWCA_with_mesh(gridPos + make_int2(grid_x,grid_y), index, size1, pos1, x,y, mesh_begin, mesh_end,atoms_id,L,cutoff,grid_num);
		}
	}

    fx[index] = force.x;
    fy[index] = force.y;
}
*/

#endif
