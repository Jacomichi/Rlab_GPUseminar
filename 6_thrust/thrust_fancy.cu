#include "thrust_all.cuh"

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

//AoS
struct Affine_float2{
  float gamma;

  Affine_float2(float _gamma){gamma = _gamma;}
  __host__ __device__
  float2 operator()(float2 r){
    float x = r.x;
    float y = r.y;

    float affine_x = x + gamma * y;
    return make_float2(affine_x,y);
  }
};

//SoA
struct Affine{
  float gamma;

  Affine(float _gamma){gamma = _gamma;}

  typedef thrust::tuple<float,float> tfloat2;
  __host__ __device__
  tfloat2 operator()(tfloat2 r){
    float x = thrust::get<0>(r);
    float y = thrust::get<1>(r);
    float affine_x = x + gamma * y;
    return thrust::make_tuple(affine_x,y);
  }

};

int main(void)
{
  constexpr int N = 1<<2;

  /*AoS version
  thrust::device_vector<float2> r(N);
  thrust::host_vector<float2> hr(N);
  float counter = 0.0f;
  for(int i = 0,last = r.size();i != last;++i){
    r[i] = make_float2(0.0f,counter);
    counter++;
  }
  Affine_float2 aff(1.0f);
  thrust::transform(r.begin(),r.end(),r.begin(),aff);

  thrust::copy(r.begin(),r.end(),hr.begin());

  for(int i = 0,last = hr.size();i != last;++i){
    std::cout << hr[i].x << " " << hr[i].y << '\n';
  }
  */

  //SoA version
  thrust::device_vector<float> x(N),y(N);
  thrust::host_vector<float> hx(N),hy(N);
  float counter = 0.0f;

  for(int i = 0,last = x.size();i != last;++i){
    x[i] = 0.0f;
    y[i] = counter;
    counter++;
  }

  Affine affine(1.0f);
  auto first = thrust::make_zip_iterator(thrust::make_tuple(x.begin(),y.begin()));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(x.end(),y.end()));

  transform(first,last,first,affine);

  thrust::copy(x.begin(),x.end(),hx.begin());
  thrust::copy(y.begin(),y.end(),hy.begin());

  for(int i = 0,last = hx.size();i != last;++i){
    std::cout << hx[i] << " " << hy[i] << '\n';
  }


  return 0;
}
