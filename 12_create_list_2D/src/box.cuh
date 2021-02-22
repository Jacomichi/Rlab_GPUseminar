#ifndef BOX_CUH
#define BOX_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"
#include "./particle.cuh"


struct Box{
  double L;
  int N;
  double rho;

  Box(int _N,double _rho){
    N = _N;
    rho = _rho;
    L = sqrt(N/rho);
  }
  
};

#endif
