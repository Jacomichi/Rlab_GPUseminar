#include "thrust_all.cuh"

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

struct saxpy_functor
{
    const float a;
    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(const float& x, const float& y) const {
      return a * x + y;
    }
};

void saxpy_fast(float a, thrust::device_vector<float>& X,thrust::device_vector<float>& Y){
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(),[=]__device__(auto _x,auto _y){return a*_x + _y;});
}


void saxpy_slow(float a, thrust::device_vector<float>& X, thrust::device_vector<float>& Y){
  thrust::device_vector<float> tmp(X.size());
  transform(X.begin(),X.end(),tmp.begin(),[=]__device__(auto _x){return a*_x;});
  thrust::transform(tmp.begin(), tmp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

int main(void)
{
  constexpr int N = 1<<10;
  using vec_type = float;
  thrust::device_vector<vec_type> x(N);
  thrust::device_vector<vec_type> y(N);
  thrust::host_vector<vec_type> z(N);
  thrust::fill(x.begin(),x.end(),1.0f);
  thrust::fill(y.begin(),y.end(),2.0f);

  //saxpy_fast(2.0f,x,y);
  saxpy_slow(2.0f,x,y);

  thrust::copy(y.begin(),y.end(),z.begin());

  /*
  for(auto i = z.begin(),last = z.end(); i!= last;++i){
    std::cout << *i << '\n';
  }
  */

  return 0;
}
