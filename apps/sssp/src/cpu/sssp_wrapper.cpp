#include <stdio.h>
#include "usssp.h"

#define INF 1073741824	// 1024*1024*1024

#include "sssp_kernel.cpp"

//#define CONCURRENT_STREAM

//unsigned int queue_max_length = QMAXLENGTH;
//unsigned int queue_length = 0;

unsigned int bSize = 0;
unsigned int iteration = 0;

void prepare_cpu()
{
	omp_set_dynamic(0);
	omp_set_num_threads(config.thread_num);
}

void clean_gpu()
{

}

void SSSP_CPU( ) 
{
	prepare_cpu();
	iteration = 0;
	start_time = gettime();
	switch (config.solution) {
		case 0:	usssp_cpu();
			break;
		default: 
			break;
	}
	end_time = gettime();
	ker_exe_time += end_time - start_time;
	
	//fprintf(stderr, "SSSP iteration:\t\t%lf\n",end_time-start_time);
	if (DEBUG)
		fprintf(stderr, "Iteration: %d\n", iteration);
	clean_gpu();
}
