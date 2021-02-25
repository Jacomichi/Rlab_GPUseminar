#ifndef SETTING_CUH
#define SETTING_CUH

#include "../thrust_all.cuh"
#include "./../utility/random.cuh"

struct Setting{
  double L;
  int N;
  double rho;
  double dt;
  int numThreads;
  int numBlocks;

  Setting(int _N, double _rho,double _dt){
    N = _N;
    rho = _rho;
    L = sqrt(N/rho);
    dt = _dt;
    numThreads = 1<<10;
    numBlocks = (N + numThreads - 1)/numThreads;
  }

};

#endif
