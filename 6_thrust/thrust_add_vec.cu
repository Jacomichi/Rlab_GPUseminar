#include "thrust_all.cuh"

__global__
void device_add(int *a,int *b,int N)
{
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  auto stride = blockDim.x * gridDim.x;
  for (auto i = index; i < N; i += stride)
    b[i] = a[i] + b[i];
}

int main(void)
{
  constexpr int N = 1<<10;
  using vec_type = int;
  thrust::device_vector<vec_type> a(N);
  thrust::device_vector<vec_type> b(N);
  thrust::host_vector<vec_type> c(N);
  thrust::fill(a.begin(),a.end(),1);
  thrust::fill(b.begin(),b.end(),2);

  //auto add_vec = thrust::plus<vec_type>();
  //thrust::transform(a.begin(),a.end(),b.begin(),b.begin(),add_vec);
  int *a_ptr =thrust::raw_pointer_cast(&a[0]);
  int *b_ptr =thrust::raw_pointer_cast(&b[0]);
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  device_add<<<numBlocks,blockSize>>>(a_ptr,b_ptr,N);

  thrust::copy(b.begin(),b.end(),c.begin());
  for(auto i = c.begin(),last = c.end(); i!= last;++i){
    std::cout << *i << '\n';
  }

  return 0;
}
