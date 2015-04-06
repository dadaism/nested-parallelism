#include <stdio.h>
#include <cuda.h>
#include "np_synth.h"

#define QMAXLENGTH 102400000
#define GM_BUFF_SIZE 102400000

#include "np_synth_kernel.cu"

dim3 dimGrid(1,1,1);	// thread+bitmap
dim3 dimBlock(1,1,1);	
int maxDegreeT = 192;	// thread/block, thread+queue
dim3 dimGridT(1,1,1);
dim3 dimBlockT(maxDegreeT,1,1);

int maxDegreeB = 32;
dim3 dimBGrid(1,1,1);	// block+bitmap
dim3 dimBBlock(maxDegreeB,1,1);		
dim3 dimGridB(1,1,1);
dim3 dimBlockB(maxDegreeB,1,1); // block+queue

//char *update = new char [data_size] ();
//int *queue = new int [queue_max_length];
unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;
unsigned int nonstop = 0;


int *d_iter;
FLOAT_T *d_rst;
FLOAT_T *d_data;

double start_time, end_time;
	
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
	init_time += end_time-start_time;
	
	start_time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(config.device_num) );
	end_time = gettime();
	if (DEBUG) {
		fprintf(stderr, "Choose CUDA device: %d\n", config.device_num);
		//fprintf(stderr, "cudaSetDevice:\t\t%lf\n",end_time-start_time);
	}
	/* Configuration for thread+bitmap*/	
	if ( data_size > maxDegreeT ){
		dimGrid.x = data_size / maxDegreeT + 1;
		dimBlock.x = maxDegreeT;
	}
	else {
		dimGrid.x = 1;
		dimBlock.x = 32 * (data_size/32+1);

	}
	/* Configuration for block+bitmap */
	if ( data_size > MAXDIMGRID ){
		dimBGrid.x = MAXDIMGRID;
		dimBGrid.y = data_size / MAXDIMGRID + 1;
	}
	else {
		dimBGrid.x = data_size;
	}
	
	/* Allocate GPU memory */
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_iter, sizeof(int)*(data_size+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_data, sizeof(FLOAT_T)*data_size ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_rst, sizeof(FLOAT_T)*data_size ) );
	end_time = gettime();
	d_malloc_time += end_time-start_time;

	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_iter, iter, sizeof(int)*(data_size+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_data, data, sizeof(FLOAT_T)*data_size, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemset( d_rst, 0, sizeof(FLOAT_T)*data_size) );
	end_time = gettime();
	h2d_memcpy_time += end_time-start_time;
}

void clean_gpu()
{
	cudaCheckError( __LINE__, cudaFree(d_iter) );
	cudaCheckError( __LINE__, cudaFree(d_data) );
	cudaCheckError( __LINE__, cudaFree(d_rst) );
}

void np_baseline_gpu()
{	
	np_thread_kernel<<<dimGrid, dimBlock>>>(d_iter, d_data, d_rst, data_size);
}

void np_dual_queue_gpu()
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

	cudaCheckError( __LINE__, cudaMemset(d_queue_length_l, 0, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMemset(d_queue_length_h, 0, sizeof(unsigned int) ) );
	/* initialize the dual working set */
	gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_iter, data_size,
														d_work_queue_l, d_queue_length_l, queue_max_length_l,
														d_work_queue_h, d_queue_length_h, queue_max_length_h );

	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_l, d_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_h, d_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	
	if (DEBUG) {
		fprintf(stderr, "Queue size of low outdegree:\t%d\n", queue_length_l);
		fprintf(stderr, "Queue size of high outdegree:\t%d\n", queue_length_h);
	}
	if ( queue_length_l!=0 ) {
		if ( queue_length_l<=maxDegreeT ) {
			dimGridT.x = 1;
		}
		else if ( queue_length_l<=maxDegreeT*MAXDIMGRID ) {
			dimGridT.x = queue_length_l / maxDegreeT + 1;
		}
		else {
			fprintf(stderr, "Too many elements in queue\n");
			exit(0);
		}

		np_thread_queue_kernel<<<dimGridT, dimBlockT>>>(d_iter, d_data, d_rst, d_work_queue_l,
														d_queue_length_l, data_size);
	}

	if ( queue_length_h!=0 ) {
		if ( queue_length_h<=MAXDIMGRID ) {
			dimGridB.x = queue_length_h;
		}
		else if ( queue_length_h<=MAXDIMGRID*1024 ) {
			dimGridB.x = MAXDIMGRID;
			dimGridB.y = queue_length_h/MAXDIMGRID+1;
		}
		np_block_queue_kernel<<<dimGridB, dimBlockB>>>(d_iter, d_data, d_rst, d_work_queue_h,
														d_queue_length_h);
	}
	cudaCheckError( __LINE__, cudaGetLastError() );
	cudaFree(d_work_queue_l);
	cudaFree(d_queue_length_l);
	cudaFree(d_work_queue_h);
	cudaFree(d_queue_length_h);
}

void np_shared_delayed_buffer_gpu()
{
	np_shared_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(d_iter, d_data, d_rst, data_size);
	cudaCheckError( __LINE__, cudaGetLastError() );
}

void np_global_delayed_buffer_gpu()
{
	unsigned int buffer_size = 0;
	unsigned int *d_buffer_size;
	int *d_buffer;
	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );

	np_global_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(d_iter, d_data, d_rst, d_buffer, d_buffer_size, data_size);
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
		
	np_block_queue_kernel<<<dimGridB, dimBlockB>>>(d_iter, d_data, d_rst, d_buffer, d_buffer_size);
            
	cudaFree(d_buffer);
	cudaFree(d_buffer_size);
}

void np_np_naive_gpu()
{
	np_multidp_kernel<<<dimGrid, dimBlock>>>(d_iter, d_data, d_rst, data_size);
	cudaCheckError( __LINE__, cudaGetLastError() );
}

void np_np_opt_gpu()
{
	int *d_buffer;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	np_singledp_kernel<<<dimGrid, dimBlock>>>(d_iter, d_data, d_rst, d_buffer, data_size);
	cudaCheckError( __LINE__, cudaGetLastError() );
	cudaFree(d_buffer);
}

void NP_SYNTH_GPU()
{
	prepare_gpu();
	
	start_time = gettime();
	switch (config.solution) {
		case 0:  np_baseline_gpu();	// 
			break;
		case 1:  np_dual_queue_gpu();	//
			break;
		case 2:  np_shared_delayed_buffer_gpu();	//
			break;
		case 3:  np_global_delayed_buffer_gpu();	//
			break;
		case 4:  np_np_naive_gpu();	//
			break;
		case 5:  np_np_opt_gpu();	//
			break;
		default:
			break;
	}
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	end_time = gettime();
	ker_exe_time += end_time-start_time;
 	
	start_time = end_time;
    cudaCheckError( __LINE__, cudaMemcpy( rst, d_rst, sizeof(FLOAT_T)*data_size, cudaMemcpyDeviceToHost) );
    end_time = gettime();
    d2h_memcpy_time += end_time-start_time;

#ifdef GPU_PROFILE
	gpu_statistics<<<1, 1>>>(config.solution);
#endif

	clean_gpu();
}
