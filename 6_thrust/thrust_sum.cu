#include "thrust_all.cuh"

int main(void)
{
  srand(time(NULL));
  thrust::host_vector<int> h_vec(1000);
  thrust::generate(h_vec.begin(),h_vec.end(),rand);
  thrust::device_vector<int> d_vec = h_vec;

  long int h_sum = thrust::reduce(h_vec.begin(),h_vec.end());
  std::cout << "host : " << h_sum << '\n';

  long int d_sum = thrust::reduce(d_vec.begin(),d_vec.end());
  std::cout << "host : " << d_sum << '\n';

  return 0;
}
