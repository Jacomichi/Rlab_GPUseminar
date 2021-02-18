#include "thrust_all.cuh"

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

int main(void)
{
  constexpr int N = 1<<10;
  using vec_type = float;
  thrust::device_vector<vec_type> x(N);
  thrust::device_vector<vec_type> y(N);
  thrust::host_vector<vec_type> z(N);
  thrust::fill(x.begin(),x.end(),1.0f);
  thrust::fill(y.begin(),y.end(),0.0f);

  auto lambda = []__device__(auto k){return -k;};
  transform(x.begin(),x.end(),y.begin(),lambda);

  thrust::copy(y.begin(),y.end(),z.begin());

  for(auto i = z.begin(),last = z.end(); i!= last;++i){
    std::cout << *i << '\n';
  }


  return 0;
}
