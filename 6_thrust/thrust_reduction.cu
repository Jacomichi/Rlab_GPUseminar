#include "thrust_all.cuh"

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

int main(void)
{
  constexpr int N = 1<<10;
  using vec_type = float;
  thrust::device_vector<vec_type> x(N);
  thrust::host_vector<vec_type> z(N);
  thrust::sequence(x.begin(),x.end());

  vec_type init = 0.0f;
  int sum = thrust::reduce(x.begin(),x.end(),init,thrust::maximum<vec_type>());

  std::cout << sum << '\n';

  return 0;
}
