#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"

#include <typeinfo>

struct size_t2{
    thrust::device_ptr<double> x;
    thrust::device_ptr<double> y;
    size_t2(thrust::device_ptr<double> _x,thrust::device_ptr<double> _y){
        x =_x;
        y =_y;
    }
};

struct size_t3{
    thrust::device_ptr<double> x;
    thrust::device_ptr<double> y;
    thrust::device_ptr<double> z;
    size_t3(thrust::device_ptr<double> _x,thrust::device_ptr<double> _y,thrust::device_ptr<double> _z){
        x = _x;
        y = _y;
        z = _z;
    }
};

void d_copy(thrust::device_ptr<double> src,thrust::device_ptr<double> dst,int N){
    thrust::copy(src,src + N,dst);
}

struct Atoms{
    unsigned int N;
    thrust::device_ptr<double> x;
    thrust::device_ptr<double> y;
    thrust::device_ptr<double> z;
    thrust::device_ptr<double> vx;
    thrust::device_ptr<double> vy;
    thrust::device_ptr<double> vz;
    thrust::device_ptr<double> random_fx;
    thrust::device_ptr<double> random_fy;
    thrust::device_ptr<double> random_fz;

    Atoms(size_t _N){
        N = _N;
        x = thrust::device_malloc<double>(_N);
        y = thrust::device_malloc<double>(_N);
        z = thrust::device_malloc<double>(_N);
        vx = thrust::device_malloc<double>(_N);
        vy = thrust::device_malloc<double>(_N);
        vz = thrust::device_malloc<double>(_N);
        random_fx = thrust::device_malloc<double>(_N);
        random_fy = thrust::device_malloc<double>(_N);
        random_fz = thrust::device_malloc<double>(_N);
    }

    //コピーコンストラクタ

    Atoms(const Atoms &_atoms){
        N = _atoms.N;
        x = thrust::device_malloc<double>(N);
        y = thrust::device_malloc<double>(N);
        z = thrust::device_malloc<double>(N);
        vx = thrust::device_malloc<double>(N);
        vy = thrust::device_malloc<double>(N);
        vz = thrust::device_malloc<double>(N);
        random_fx = thrust::device_malloc<double>(N);
        random_fy = thrust::device_malloc<double>(N);
        random_fz = thrust::device_malloc<double>(N);
        d_copy(_atoms.x,x,N);
        d_copy(_atoms.y,y,N);
        d_copy(_atoms.z,z,N);
        d_copy(_atoms.vx,vx,N);
        d_copy(_atoms.vy,vy,N);
        d_copy(_atoms.vz,vz,N);
        d_copy(_atoms.random_fx,random_fx,N);
        d_copy(_atoms.random_fy,random_fy,N);
        d_copy(_atoms.random_fz,random_fz,N);
    };

    Atoms &operator=(const Atoms &_atoms) {
        if(this != &_atoms) {
            this->N = _atoms.N;
            d_copy(_atoms.x,this->x,N);
            d_copy(_atoms.y,this->y,N);
            d_copy(_atoms.z,this->z,N);
            d_copy(_atoms.vx,this->vx,N);
            d_copy(_atoms.vy,this->vy,N);
            d_copy(_atoms.vz,this->vz,N);
            d_copy(_atoms.random_fx,this->random_fx,N);
            d_copy(_atoms.random_fy,this->random_fy,N);
            d_copy(_atoms.random_fz,this->random_fz,N);
        }
        return *this;
    }


    ~Atoms(){
        thrust::device_free(x);
        thrust::device_free(y);
        thrust::device_free(z);
        thrust::device_free(vx);
        thrust::device_free(vy);
        thrust::device_free(vz);
        thrust::device_free(random_fx);
        thrust::device_free(random_fy);
        thrust::device_free(random_fz);
    }

    void random_uniform(curandGenerator_t gen){
        curandGenerateUniformDouble(gen,thrust::raw_pointer_cast(x),N);
        curandGenerateUniformDouble(gen,thrust::raw_pointer_cast(y),N);
        curandGenerateUniformDouble(gen,thrust::raw_pointer_cast(z),N);
    }

    void create_thermal_noise(curandGenerator_t gen,double noise_strength){
        curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(random_fx),N,0.0,noise_strength);
        curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(random_fy),N,0.0,noise_strength);
        curandGenerateNormalDouble(gen,thrust::raw_pointer_cast(random_fz),N,0.0,noise_strength);
    }

    size_t3 end(){
        return size_t3(x+N,y+N,z+N);
    }

    size_t3 first(){
        return size_t3(x,y,z);
    }

    auto first_zip_iter(){
        return thrust::make_zip_iterator(thrust::make_tuple(x,y,z));
    }

    auto end_zip_iter(){
        return thrust::make_zip_iterator(thrust::make_tuple(x+N,y+N,z+N));
    }

    void updatePos(double dt){
        //TO DO : using async transform makes this function much faster probably
        auto update_position = [=]__device__(auto _r,auto _v){return _r + _v*dt;};
        thrust::transform(x,x+N,vx,x,update_position);
        thrust::transform(y,y+N,vy,y,update_position);
        thrust::transform(z,z+N,vz,z,update_position);
    }

    void updateVelo(curandGenerator_t gen,double dt,double mass){
        double noise_strength = sqrt(2.0 * dt);
        create_thermal_noise(gen,noise_strength);
        auto update_velocity = [=]__device__(auto _v,auto _rnd){
            return _v * (1.0 - dt/mass) + _rnd/mass;};
        thrust::transform(vx,vx+N,random_fx,vx,update_velocity);
        thrust::transform(vy,vy+N,random_fy,vy,update_velocity);
        thrust::transform(vz,vz+N,random_fz,vz,update_velocity);
    }


    void fillPos(double val){
        thrust::fill(x,x+N,val);
        thrust::fill(y,y+N,val);
        thrust::fill(z,z+N,val);
    }

    void fillVelo(double val){
        thrust::fill(vx,vx+N,val);
        thrust::fill(vy,vy+N,val);
        thrust::fill(vz,vz+N,val);
    }

    double ave_v_sq(){
        double vx_sq = thrust::inner_product(vx, vx + N, vx, 0.0);
        double vy_sq = thrust::inner_product(vy, vy + N, vy, 0.0);
        double vz_sq = thrust::inner_product(vz, vz + N, vz, 0.0);

        return (vx_sq + vy_sq + vz_sq)/(3.0 * static_cast<double>(N));
    }
};



struct inner{
    typedef thrust::tuple<double,double,double> tuple_double3;
    //typedef thrust::tuple<double,double,double,double,double,double> tuple_double6;
    __host__ __device__
    double operator()(const tuple_double3& r){
        printf( "in\n" );
        auto _x = thrust::get<0>(r);
        auto _y = thrust::get<1>(r);
        auto _z = thrust::get<2>(r);

        return (_x * _x + _y * _y + _z * _z);
    }
};

struct distance{
    typedef thrust::tuple<double,double,double,double,double,double> tuple_double6;
    __host__ __device__
    double operator()(const tuple_double6 &r){
        double _x1 = thrust::get<0>(r);
        double _y1 = thrust::get<1>(r);
        double _z1 = thrust::get<2>(r);
        double _x2 = thrust::get<3>(r);
        double _y2 = thrust::get<4>(r);
        double _z2 = thrust::get<5>(r);
        double disp = ((_x1 - _x2) * (_x1 - _x2) + (_y1 - _y2)*(_y1 - _y2) + (_z1 - _z2)*(_z1 - _z2));
        return disp;
    }
};

double ave_dist(const Atoms &atoms1,const Atoms &atoms2){
    size_t N = atoms1.N;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(atoms1.x,atoms1.y ,atoms1.z,atoms2.x,atoms2.y ,atoms2.z));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(atoms1.x + N,atoms1.y + N ,atoms1.z + N,atoms2.x + N ,atoms2.y + N ,atoms2.z + N));

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

struct _dist3D{
    typedef thrust::tuple<double,double,double> tuple_double3;
    __host__ __device__
    double operator()(const tuple_double3& r){
        double x = thrust::get<0>(r);
        double y = thrust::get<1>(r);
        double z = thrust::get<2>(r);
        return (x * x + y * y + z * z);
    }
};

double dist3D(const Atoms &atoms){
    int N = atoms.N;

    auto first = thrust::make_zip_iterator(thrust::make_tuple(atoms.x,atoms.y,atoms.z));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(atoms.x + N,atoms.y + N,atoms.z + N));
    double disp = thrust::transform_reduce(first,last,_dist3D(),0.0,thrust::plus<double>());

    return disp/(static_cast<double>(N));
}


#endif
