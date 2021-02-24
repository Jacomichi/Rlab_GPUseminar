#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include "./../thrust_all.cuh"
#include "./../utility/random.cuh"
#include "./setting.cuh"
#include "./structs.h"

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


struct update_velocity{
    double dt;
    double friction;
    double strength;

    update_velocity(double _dt,double _friction,double _strength){
        dt = _dt;
        friction = _friction;
        strength = _strength;
    }

    typedef thrust::tuple<double,double,double> tuple_double3;
    __host__ __device__
    double operator()(const tuple_double3 &r){
        double _v = thrust::get<0>(r);
        double _f = thrust::get<1>(r);
        double _noise = thrust::get<2>(r);
        return _v * (1.0 - dt * friction) + _f * dt + strength * _noise;
    }
};

struct L2norm{
    typedef thrust::tuple<double,double> tuple_double2;
    __host__ __device__
    double operator()(const tuple_double2 &r){
        double _x = thrust::get<0>(r);
        double _y = thrust::get<1>(r);
        return _x * _x + _y * _y;
    }
};

struct update_velocity_quench{
    double dt;
    double friction;

    update_velocity_quench(double _dt,double _friction){
        dt = _dt;
        friction = _friction;
    }

    typedef thrust::tuple<double,double> tuple_double2;
    __host__ __device__
    double operator()(const tuple_double2 &r){
        double _v = thrust::get<0>(r);
        double _f = thrust::get<1>(r);
        return _v * (1.0 - dt * friction) + _f * dt;
    }
};

struct Atoms{
    unsigned int N;
    double rho;
    double temperature;
    double friction;
    double large_sigma;
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

    Atoms(size_t _N,double _rho,double _temperature,double _friction, double _large_sigma){
        N = _N;
        rho = _rho;
        temperature = _temperature;
        friction = _friction;
        large_sigma = _large_sigma;
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
        rho = _atoms.rho;
        temperature = _atoms.temperature;
        friction = _atoms.friction;
        large_sigma = _atoms.large_sigma;
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
            this->rho = _atoms.rho;
            this->temperature = _atoms.temperature;
            this->friction = _atoms.friction;
            this->large_sigma = _atoms.large_sigma;
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


    void create_square_lattice(double L){
        //double L = box.L;
        int num_atoms_per_axis = ceil(sqrt(N));
        std::cout << "num_atoms_per_axis " << num_atoms_per_axis<< '\n';
        //端の粒子が重ならないように、系を少し小さく見積もる。
        double lattice_size = (L - 1.0) / static_cast<double>(num_atoms_per_axis);
        std::cout << "lattice size " << lattice_size<< '\n';
        thrust::counting_iterator<int> id_iter(0);
        thrust::counting_iterator<int> id_iter_end(N);
        auto y_pos = [=]__device__(int atom_id){return static_cast<double>((atom_id / num_atoms_per_axis)) * lattice_size;};
        auto x_pos = [=]__device__(int  atom_id){
            return static_cast<double>((atom_id %num_atoms_per_axis)) * lattice_size;
        };

        thrust::transform(thrust::device,id_iter,id_iter_end,x,x_pos);
        thrust::transform(thrust::device,id_iter,id_iter_end,y,y_pos);
    }

    void set_in_box(Setting box){
        double L = box.L;
      auto PBC_pos = [=]__device__(double r){
        return r - static_cast<double>(floor(r/L))*L;
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

    void updateVelo(curandGenerator_t gen,double dt){
        double noise_strength = sqrt(2.0 * friction * temperature * dt);
        create_thermal_noise(gen,1.0);
        auto velo_update = update_velocity(dt,friction,noise_strength);
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(vx,fx,random_fx)),
            thrust::make_zip_iterator(thrust::make_tuple(vx + N,fx+ N,random_fx+ N)),
            vx,velo_update
        );
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(vy,fy,random_fy)),
            thrust::make_zip_iterator(thrust::make_tuple(vy + N,fy + N,random_fy+ N)),
            vy,velo_update
        );
    }

    void updateQuench(double dt){
        auto update_position = [=]__device__(auto _r,auto _f){return _r + _f*dt;};
        thrust::transform(x,x+N,fx,x,update_position);
        thrust::transform(y,y+N,fy,y,update_position);
    }

    void monodisperse(){
        thrust::fill(sigma,sigma+N,1.0);
    }

    void binary(){
        int half = int(N/2) + 1;
        thrust::fill(sigma,sigma+half,1.0);
        thrust::fill(sigma+half,sigma + N,large_sigma);
    }

    void set_diameter(){
        if(large_sigma == 1.0){
            monodisperse();
        }else{
            binary();
        }
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

    double ave_velocity(){
        return (thrust::reduce(vx,vx + N) + thrust::reduce(vx,vx + N,0.0))/(2.0 * static_cast<double>(N));
    }

    bool is_force_balance(){
        auto first = thrust::make_zip_iterator(thrust::make_tuple(fx,fy));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(fx + N ,fy + N));

        double max_f_sq = thrust::transform_reduce(first,last,L2norm(),0.0,thrust::maximum<double>());
        if( (max_f_sq) < 1e-22){
            return true;
        }else{
            return false;
        }
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
double calcPotentialEnergyWCA(double2 pos1, double2 pos2, double size,double L) {
	double pot = 0.0;
    constexpr double cutoff =1.122462048309373;
	double2 dr = pos1 - pos2;
	dr.x = dr.x - L*floor(dr.x/L+0.5);
	dr.y = dr.y - L*floor(dr.y/L+0.5);
	double r = sqrt(dot(dr, dr));
	if(r < size*cutoff) {
        double invr = size/r;
        invr *= invr;
		double invR = invr * invr * invr;
		pot = 4*invR*(invR - 1.0) + 1.0;
	}
	return pot;
}

__device__
double calcPotentialEnergyHarmonic(double2 pos1, double2 pos2, double size,double L) {
	double pot = 0.0;
	double2 dr = pos1 - pos2;
	dr.x = dr.x - L*floor(dr.x/L+0.5);
	dr.y = dr.y - L*floor(dr.y/L+0.5);
	double r = sqrt(dot(dr, dr));
	if(r < size) {
		pot = (1.0 - r /size)*(1.0 - r /size);
	}
	return pot;
}

__device__
double calcPotentialEnergySoft(double2 pos1, double2 pos2, double size,double L) {
	double pot = 0.0;
    //constexpr double cutoff = 3.0;
	double2 dr = pos1 - pos2;
	dr.x = dr.x - L*floor(dr.x/L+0.5);
	dr.y = dr.y - L*floor(dr.y/L+0.5);
	double r = sqrt(dot(dr, dr));
    double invr = size/r;
    invr *= invr;
    double invR = invr * invr * invr;
    pot = invR * invR ;
	return pot;
}

__global__
void calc_potential(double *x,double *y, double *pot_energy,double *sigma,int N,double L) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    double pot = 0.0;

    for(int index2 = 0;index2 < N;++index2){
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
            double size = (size1 + sigma[index2])/2.0;
            pot += calcPotentialEnergySoft(pos1, pos2, size,L);
        }
    }
    pot_energy[index] = pot;
}

__global__
void calc_potential_with_list(double *x,double *y, double *pot_energy,double *sigma,double L,int *list,int list_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    double pot = 0.0;

    int particles_in_list = list[index * list_size];

    for(int i = 1; i < particles_in_list; ++i){
        int index2 = list[index * list_size + i];
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
            double size = (size1 + sigma[index2])/2.0;
            pot += calcPotentialEnergySoft(pos1, pos2, size,L);
        }
    }
    pot_energy[index] = pot;
}

__global__
void calc_potential_Harmonic(double *x,double *y, double *pot_energy,double *sigma,double L,int *list,int list_size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    double pot = 0.0;

    int particles_in_list = list[index * list_size];

    for(int i = 1; i < particles_in_list; ++i){
        int index2 = list[index * list_size + i];
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
            double size = (size1 + sigma[index2])/2.0;
            pot += calcPotentialEnergyHarmonic(pos1, pos2, size,L);
        }
    }
    pot_energy[index] = pot;
}

__device__
double2 calcForceWCA(double2 pos1, double2 pos2, double size,double L) {
	double2 force = make_double2(0.0);
    constexpr double cutoff =1.122462048309373;
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

__device__
double2 calcForceSoft(double2 pos1, double2 pos2, double size,double L) {
	double2 force = make_double2(0.0);
    constexpr double cutoff = 3.0;
	double2 dr = pos1 - pos2;
	dr.x = dr.x - L*floor(dr.x/L+0.5);
	dr.y = dr.y - L*floor(dr.y/L+0.5);
	double r = sqrt(dot(dr, dr));
	if(r < size*cutoff) {
        double invr = size/r;
        invr *= invr;
		double invR = invr * invr * invr;
		force = 12.0 * invR * invR * invr * dr;
	}
	return force;
}

__device__
double2 calcForceHarmonic(double2 pos1, double2 pos2, double size,double L) {
	double2 force = make_double2(0.0);
	double2 dr = pos1 - pos2;
	dr.x = dr.x - L*floor(dr.x/L+0.5);
	dr.y = dr.y - L*floor(dr.y/L+0.5);
	double r = sqrt(dot(dr, dr));
	if(r < size) {
		dr = dr / r;
		force = (1.0 - r/size) * dr/size;
	}
	return force;
}

__global__
void calc_force(double *x,double *y, double *fx, double *fy,double *sigma,int N,double L) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    double2 force = make_double2(0.0);

    for(int index2 = 0;index2 < N;++index2){
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
            double size = (size1 + sigma[index2])/2.0;
            force += calcForceSoft(pos1, pos2, size,L);
        }
    }
    fx[index] = force.x;
    fy[index] = force.y;
}

__global__
void calc_force_with_list(double *x,double *y, double *fx, double *fy,double *sigma,double L,int *list,int list_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    double2 force = make_double2(0.0);

    int particles_in_list = list[index * list_size];

    for(int i = 1; i < particles_in_list; ++i){
        int index2 = list[index * list_size + i];
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
            double size = (size1 + sigma[index2])/2.0;
            force += calcForceSoft(pos1, pos2, size,L);
        }
    }
    fx[index] = force.x;
    fy[index] = force.y;
}


__global__
void calc_force_Harmonic(double *x,double *y, double *fx, double *fy,double *sigma,double L,int *list,int list_size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    double2 pos1 = make_double2(x[index],y[index]);
    double size1 = sigma[index];
    double2 force = make_double2(0.0);

    int particles_in_list = list[index * list_size];

    for(int i = 1; i < particles_in_list; ++i){
        int index2 = list[index * list_size + i];
        if(index2 != index){
            double2 pos2 = make_double2(x[index2],y[index2]);
            double size = (size1 + sigma[index2])/2.0;
            force += calcForceHarmonic(pos1, pos2, size,L);
        }
    }
    fx[index] = force.x;
    fy[index] = force.y;
}


#endif
