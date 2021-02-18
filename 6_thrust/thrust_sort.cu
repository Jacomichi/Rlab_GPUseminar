#include "thrust_all.cuh"

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

int main(void)
{
  constexpr int N = 5;
  using vec_type = float;
  float keys[N] = {5.0f,3.0f,1.0f,3.0f,12.0f};
  char values[N] = {'a','d','z','k','n'};
  thrust::sort_by_key(values,values+N,keys);

  for(std::size_t i = 0; i < N; ++i){
    std::cout << "key:" << keys[i] << " value:" <<values[i] << '\n';
  }

  return 0;
}
