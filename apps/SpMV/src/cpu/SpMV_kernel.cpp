#include <stdio.h>
#include "SpMV.h"

void prepare_cpu()
{	
	/* generate random data */
	srand(time(NULL));
	data = new FLOAT_T [noEdgeTotal] ();
	x = new FLOAT_T [noNodeTotal] ();
	y = new FLOAT_T [noNodeTotal] ();
	for (int i=0; i<noEdgeTotal; ++i) 
		//data[i] = rand() % 10 + 0.5;
		data[i] = i%13 + 0.5;
	for (int i=0; i<noNodeTotal; ++i) 
		//x[i] = rand() % 10 + 0.5;
		x[i] = i%17 + 0.5;
}

void clean_cpu()
{
	delete [] data;
	delete [] x;
	delete [] y;
}

void spmv_omp()
{
	int *ptr = graph.vertexArray;
	int *indices = graph.edgeArray;
	#pragma omp parallel for
	for (int tid=0; tid<noNodeTotal; ++tid) {
		FLOAT_T dot = 0;
		/* get neighbour range */
		int start = ptr[tid];
		int end = ptr[tid+1];
		/* access neighbours */
		for (int i=start; i<end; ++i) {
			dot += data[i] * x[indices[i]];
		}
		y[tid] = dot;
	}
}


void SpMV_CPU()
{
	prepare_cpu();
	
	start_time = gettime();
	switch (config.solution) {
		case 0:  spmv_omp();	// 
			break;
		default:
			break;
	}
	end_time = gettime();
	ker_exe_time += end_time - start_time;
	start_time = end_time;
	end_time = gettime();
    d2h_memcpy_time += end_time-start_time;
	clean_cpu();
}

