#ifndef REDUCTION_CUH
#define REDUCTION_CUH

__global__
void reduction_kernel(double *output,double *input, int stride, int N){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int in_idx = idx + stride;
    if(in_idx < N){
        output[idx] += input[in_idx];
    }
}

void reduction(double *output,double *input, int num_threads, int N){
    int num_blocks = (N + num_threads - 1)/num_threads;
    for(int stride = 1; stride < N ; stride *=2){
        reduction_kernel<<<num_blocks,num_threads>>>(output,input,stride,N);
    }
}

#endif
