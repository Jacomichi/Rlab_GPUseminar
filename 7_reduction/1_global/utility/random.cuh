#ifndef RANDOM_H
#define RANDOM_H

#include <curand.h>
#include <curand_kernel.h>
#include <random>

curandGenerator_t createRandGenerator(){
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    std::random_device rnd;
    curandSetPseudoRandomGeneratorSeed(gen,rnd());
    return gen;
}

#endif
