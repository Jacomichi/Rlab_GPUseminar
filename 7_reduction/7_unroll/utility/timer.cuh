#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

class Timer{
	timespec start_time, end_time;
	double duration;

public:
	Timer():start_time((timespec){0,0}),end_time((timespec){0,0}){}

	void start_record(){ clock_gettime(CLOCK_REALTIME, &start_time); }
	void stop_record(){
    clock_gettime(CLOCK_REALTIME, &end_time);
	duration =  1.0e+3 * (double)(end_time.tv_sec - start_time.tv_sec) + (double)((end_time.tv_nsec - start_time.tv_nsec)/1000000);
  	}
	void print_result(){
		std::cout << duration << "[ms]" << '\n';
	}
};

class cudaTimer{
	cudaEvent_t start, stop;
	float elapse;

public:
	cudaTimer(){
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	~cudaTimer(){
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	void start_record(){cudaEventRecord(start);}
	void stop_record(){
		cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapse, start, stop);
	}
	void print_result(){
		std::cout<< elapse <<"[ms]"<<std::endl;
	}
};

#endif
