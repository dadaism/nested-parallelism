#include <stdio.h>
#include <cuda.h>
#include "SpMV.h"

#define QMAXLENGTH 10240000
#define GM_BUFF_SIZE 10240000

#define THREADS_PER_BLOCK 192

#ifndef CONSOLIDATE_LEVEL
#define CONSOLIDATE_LEVEL 1
#endif

#include "SpMV_kernel.cu"

int *d_vertexArray;
int *d_edgeArray;
int *d_levelArray;
int *d_work_queue;
char *d_frontier;
char *d_update;
FLOAT_T *d_data;
FLOAT_T *d_x;
FLOAT_T *d_y;

unsigned int *d_queue_length;
unsigned int *d_nonstop;

dim3 dimGrid(1,1,1);	// thread+bitmap
dim3 dimBlock(1,1,1);	
int maxDegreeT = THREADS_PER_BLOCK;	// thread/block, thread+queue
dim3 dimGridT(1,1,1);
dim3 dimBlockT(maxDegreeT,1,1);

int maxDegreeB = NESTED_BLOCK_SIZE;
dim3 dimBGrid(1,1,1);	// block+bitmap
dim3 dimBBlock(maxDegreeB,1,1);		
dim3 dimGridB(1,1,1);
dim3 dimBlockB(maxDegreeB,1,1); // block+queue

//char *update = new char [noNodeTotal] ();
//int *queue = new int [queue_max_length];
unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;
unsigned int nonstop = 0;

inline void cudaCheckError(int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: line %d %s\n", line, cudaGetErrorString(ce));
		exit(1);
	}
}

void prepare_gpu()
{	
	start_time = gettime();
	cudaFree(NULL);
	end_time = gettime();
	init_time += end_time - start_time;

	start_time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(config.device_num) );
	end_time = gettime();
	if (DEBUG) {
		fprintf(stderr, "Choose CUDA device: %d\n", config.device_num);
		fprintf(stderr, "cudaSetDevice:\t\t%lf\n",end_time-start_time);
	}
	/* Configuration for thread+bitmap*/	
	if ( noNodeTotal > maxDegreeT ){
		dimGrid.x = noNodeTotal / maxDegreeT + 1;
		dimBlock.x = maxDegreeT;
	}
	else {
		dimGrid.x = 1;
		dimBlock.x = 32*(noNodeTotal/32+1);
	}
	/* Configuration for block+bitmap */
	if ( noNodeTotal > MAXDIMGRID ){
		dimBGrid.x = MAXDIMGRID;
		dimBGrid.y = noNodeTotal / MAXDIMGRID + 1;
	}
	else {
		dimBGrid.x = noNodeTotal;
	}
	
	/* Allocate GPU memory */
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_update, sizeof(char)*(noNodeTotal) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_data, sizeof(FLOAT_T)*(noEdgeTotal) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_x, sizeof(FLOAT_T)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_y, sizeof(FLOAT_T)*noNodeTotal ) );
	
	end_time = gettime();
	d_malloc_time += end_time - start_time;
	
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
	

	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_data, data, sizeof(FLOAT_T)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_x, x, sizeof(FLOAT_T)*noNodeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemset(d_y, 0, sizeof(FLOAT_T)*noNodeTotal) );
	end_time = gettime();
	h2d_memcpy_time += end_time-start_time;
}

void clean_gpu()
{
	cudaCheckError( __LINE__, cudaFree(d_vertexArray) );
	cudaCheckError( __LINE__, cudaFree(d_edgeArray) );
	cudaCheckError( __LINE__, cudaFree(d_update) );
	cudaCheckError( __LINE__, cudaFree(d_data) );
	cudaCheckError( __LINE__, cudaFree(d_x) );
	cudaCheckError( __LINE__, cudaFree(d_y) );
}

void spmv_gpu()
{
	/* prepare GPU */

	csr_spmv_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray,
											d_data, d_x, d_y, noNodeTotal);
}

void spmv_dual_queue_gpu()
{
	unsigned int queue_max_length_l = QMAXLENGTH*4/5;
	unsigned int queue_max_length_h = QMAXLENGTH*1/5;
	int queue_length_l;
	int queue_length_h;
	int *d_work_queue_l;
	int *d_work_queue_h;
	unsigned int *d_queue_length_l;
	unsigned int *d_queue_length_h;
	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue_l, sizeof(int)*queue_max_length_l ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue_h, sizeof(int)*queue_max_length_h ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length_l, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length_h, sizeof(unsigned int) ) );

	cudaCheckError( __LINE__, cudaMemset( d_queue_length_l, 0, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMemset( d_queue_length_h, 0, sizeof(unsigned int) ) );
	
	/* initialize the dual working set */
	cudaCheckError( __LINE__, cudaMemset(d_update, 1, sizeof(char)*noNodeTotal) );
	gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_update, noNodeTotal,
															d_work_queue_l, d_queue_length_l, queue_max_length_l,
															d_work_queue_h, d_queue_length_h, queue_max_length_h );
	
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_l, d_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_h, d_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	
	if (DEBUG) {
		fprintf(stderr, "Queue size of low outdegree:\t%d\n", queue_length_l);
		fprintf(stderr, "Queue size of high outdegree:\t%d\n", queue_length_h);
	}

	if ( queue_length_l!=0 ) {
		if ( queue_length_l<=maxDegreeT )	{
			dimGridT.x = 1;
		}
		else if ( queue_length_l<=maxDegreeT*MAXDIMGRID ) {
			dimGridT.x = queue_length_l / maxDegreeT + 1;
		}
		else {
			fprintf(stderr, "Too many elements in queue\n");
			exit(0);
		}

		csr_spmv_thread_queue_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_work_queue_l,
															d_queue_length_l, d_data, d_x, d_y, noNodeTotal);
	}
	if ( queue_length_h!=0 ) {
		if ( queue_length_h<=MAXDIMGRID ) {
			dimGridB.x = queue_length_h;
		}
		else if ( queue_length_h<=MAXDIMGRID*1024 ) {
			dimGridB.x = MAXDIMGRID;
			dimGridB.y = queue_length_h/MAXDIMGRID+1;
		}
		csr_spmv_block_queue_kernel<<<dimGridB, dimBlockB>>>(d_vertexArray, d_edgeArray, d_work_queue_h, 
															d_queue_length_h, d_data, d_x, d_y, noNodeTotal );
	}
	cudaCheckError( __LINE__, cudaGetLastError() );

	cudaFree(d_work_queue_l);
	cudaFree(d_queue_length_l);
	cudaFree(d_work_queue_h);
	cudaFree(d_queue_length_h);
}

void spmv_shared_delayed_buffer_gpu()
{

	csr_spmv_shared_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_data,
																	d_x, d_y, noNodeTotal);
	cudaCheckError( __LINE__, cudaGetLastError() );
}

void spmv_global_delayed_buffer_gpu()
{
	unsigned int buffer_size = 0;
	unsigned int *d_buffer_size;
	int *d_buffer;
	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMemset( d_buffer_size, 0, sizeof(unsigned int) ) );

	csr_spmv_global_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_data, 
													d_x, d_y, d_buffer, d_buffer_size, noNodeTotal );
	cudaCheckError( __LINE__, cudaMemcpy(&buffer_size, d_buffer_size, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
#ifdef CPU_PROFILE
	fprintf(stderr, "Iteration %d - On CPU buffer size : %d\n", dist, buffer_size);
		//fprintf(stderr, "%d\t%d\n", dist, buffer_size);
#endif
	if (buffer_size<=MAXDIMGRID){
		dimGridB.x = buffer_size;                                                                     
	}
	else if (buffer_size<=MAXDIMGRID*1024){                               
		dimGridB.x = MAXDIMGRID;
		dimGridB.y = buffer_size/MAXDIMGRID+1;
	}                   
	else {
		fprintf(stderr, "Too many elements in queue\n");
		exit(0);        
	}
		
	csr_spmv_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_buffer, d_buffer_size,
															d_data, d_x, d_y, noNodeTotal );
	cudaFree(d_buffer);
	cudaFree(d_buffer_size);
}

void spmv_np_naive_gpu()
{
	csr_spmv_multidp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_data,
													d_x, d_y, noNodeTotal);
	cudaCheckError( __LINE__, cudaGetLastError() );
}

void spmv_np_opt_gpu()
{
	int *d_buffer;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	csr_spmv_singledp_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_data, d_x, d_y,
													noNodeTotal, d_buffer);
	cudaCheckError( __LINE__, cudaGetLastError() );
	cudaFree(d_buffer);
}

void spmv_np_consolidate_gpu()
{
	int *d_buffer;
	unsigned int *d_buffer_size;
	unsigned int *d_count;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_count, sizeof(unsigned int) ) );

#if (CONSOLIDATE_LEVEL==0)
	csr_spmv_warp_dp_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_data, d_x, d_y,
													noNodeTotal, d_buffer);
#elif (CONSOLIDATE_LEVEL==1)
	csr_spmv_block_dp_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_data, d_x, d_y,
													noNodeTotal, d_buffer);
#elif (CONSOLIDATE_LEVEL==2)
	cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)));
	cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
	csr_spmv_grid_dp_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_data, d_x, d_y,
													noNodeTotal, d_buffer, d_buffer_size, d_count);
#endif
	cudaCheckError( __LINE__, cudaGetLastError() );
	cudaFree(d_buffer);
}


void SpMV_GPU()
{
	prepare_gpu();
	
	start_time = gettime();
	switch (config.solution) {
		case 0:  spmv_gpu();	// 
			break;
		case 1:  spmv_dual_queue_gpu();	//
			break;
		case 2:  spmv_shared_delayed_buffer_gpu();	//
			break;
		case 3:  spmv_global_delayed_buffer_gpu();	//
			break;
		case 4:  spmv_np_naive_gpu();	//
			break;
		case 5:  spmv_np_opt_gpu();	//
			break;
		case 6:  spmv_np_consolidate_gpu();	//
					break;
		default:
			break;
	}
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	end_time = gettime();
	ker_exe_time += end_time - start_time;
	start_time = end_time;
	cudaCheckError( __LINE__, cudaMemcpy( y, d_y, sizeof(FLOAT_T)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime();
    d2h_memcpy_time += end_time-start_time;
#ifdef GPU_PROFILE
        gpu_statistics<<<1, 1>>>(config.solution);
#endif
	clean_gpu();
}

