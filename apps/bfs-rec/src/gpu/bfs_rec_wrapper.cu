#include <stdio.h>
#include <cuda.h>
#include "bfs_rec.h"

#define QMAXLENGTH 10240000
#define GM_BUFF_SIZE 10240000

#ifndef THREADS_PER_BLOCK_FLAT	//block size for flat parallelism
#define THREADS_PER_BLOCK_FLAT 256
#endif

#ifndef NUM_BLOCKS_FLAT
#define NUM_BLOCKS_FLAT 256
#endif

#ifndef CONSOLIDATE_LEVEL
#define CONSOLIDATE_LEVEL 1
#endif

#include "bfs_rec_kernel.cu"

int *d_vertexArray;
int *d_edgeArray;
int *d_levelArray;
int *d_work_queue;
char *d_frontier;
char *d_update;

unsigned int *d_queue_length;
unsigned int *d_nonstop;

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

//char *update = new char [noNodeTotal] ();
//int *queue = new int [queue_max_length];
unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;
unsigned int nonstop = 0;

double start_time, end_time;
	
inline void cudaCheckError(char* file, int line, cudaError_t ce)
{
	if (ce != cudaSuccess){
		printf("Error: file %s, line %d %s\n", file, line, cudaGetErrorString(ce));
		exit(1);
	}
}

void prepare_gpu()
{	
	start_time = gettime();
	cudaFree(NULL);
	end_time = gettime();
	if (VERBOSE) {
		fprintf(stderr, "CUDA runtime initialization:\t\t%lf\n",end_time-start_time);
	}
	start_time = gettime();
	cudaCheckError( __FILE__, __LINE__, cudaSetDevice(config.device_num) );
	end_time = gettime();
	if (VERBOSE) {
		fprintf(stderr, "Choose CUDA device: %d\n", config.device_num);
		fprintf(stderr, "cudaSetDevice:\t\t%lf\n",end_time-start_time);
	}
	
	/* Allocate GPU memory */
	start_time = gettime();
	cudaCheckError( __FILE__, __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaCheckError( __FILE__, __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
	cudaCheckError( __FILE__, __LINE__, cudaMalloc( (void**)&d_levelArray, sizeof(int)*noNodeTotal ) );
	//cudaCheckError( __LINE__, cudaMalloc( (void**)&d_nonstop, sizeof(unsigned int) ) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "GPU allocation time = \t\t%lf\n",end_time-start_time);

	start_time = gettime();
	cudaCheckError( __FILE__, __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __FILE__, __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "mem copy to GPU time = \t\t%lf\n", end_time-start_time);
}

void clean_gpu()
{
	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_levelArray);
}

// ----------------------------------------------------------
// version #1 - flat parallelism - level-based BFS traversal
// ----------------------------------------------------------

void bfs_flat_gpu()
{	
	/* prepare GPU */

	//copy the level array from CPU to GPU	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(unsigned )*noNodeTotal, cudaMemcpyHostToDevice) );

	bool queue_empty = false;
	bool *d_queue_empty;
	
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_queue_empty, sizeof(bool)) );

	unsigned level = 0;	

	start_time = gettime_ms();
	//level-based traversal
	while (!queue_empty){
		cudaCheckError(  __FILE__, __LINE__, cudaMemset( d_queue_empty, true, sizeof(bool)) );
		bfs_kernel_flat<<<NUM_BLOCKS_FLAT, THREADS_PER_BLOCK_FLAT>>>(level,noNodeTotal, d_vertexArray, d_edgeArray, d_levelArray, d_queue_empty);
		cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
		cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( &queue_empty, d_queue_empty, sizeof(bool), cudaMemcpyDeviceToHost) );
		level++;
	}

	end_time=gettime_ms(); // end timing execution
	printf("===> GPU #1 - flat parallelism: computation time = %.2f ms.\n", end_time-start_time);
	
	//copy the level array from GPU to CPU;
	start_time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph.levelArray, d_levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime_ms();
	printf("mem copy to CPU time = %.2f ms.\n", end_time-start_time);
}

// ----------------------------------------------------------
// version #2 - dynamic parallelism - naive 
// ----------------------------------------------------------
void bfs_rec_dp_naive_gpu()
{
	/* prepare GPU */
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyHostToDevice) );

	start_time = gettime_ms();
	int children = graph.vertexArray[source+1]-graph.vertexArray[source];
	unsigned block_size = min (children, THREADS_PER_BLOCK);
	bfs_kernel_dp<<<1,block_size>>>(source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	
	end_time = gettime_ms(); //end timing execution
	printf("===> GPU #2 - nested parallelism naive: computation time = %.2f s.\n", end_time-start_time);

	
	//copy the level array from GPU to CPU;
	start_time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph.levelArray, d_levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime_ms();
	printf("mem copy to CPU time = %.2f ms.\n", end_time-start_time);
}

// ----------------------------------------------------------
// version #3 - dynamic parallelism - hierarchical
// ----------------------------------------------------------
void bfs_rec_dp_hier_gpu()
{
	start_time = gettime_ms(); // start timing execution
	
	//copy the level array from CPU to GPU	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyHostToDevice) );

	//recursive BFS traversal - hierarchical
	int children = graph.vertexArray[source+1]-graph.vertexArray[source];
	bfs_kernel_dp_hier<<<children, THREADS_PER_BLOCK>>>(source, d_vertexArray, d_edgeArray, d_levelArray);
	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	printf("===> GPU #3 - nested parallelism hierarchical: computation time = %.2f ms.\n", gettime_ms()-start_time);

	//copy the level array from GPU to CPU;
	start_time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph.levelArray, d_levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyDeviceToHost) );
	printf("mem copy to CPU time = %.2f ms.\n", gettime_ms()-start_time);
}

// ----------------------------------------------------------
// version #4 - dynamic parallelism - consolidation
// ----------------------------------------------------------
void bfs_rec_dp_cons_gpu()
{
	start_time = gettime_ms(); // start timing execution
	
	//copy the level array from CPU to GPU	
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyHostToDevice) );

	//recursive BFS traversal - dynamic parallelism consolidation
	unsigned int *d_buffer;
	unsigned int *d_idx;
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_buffer, sizeof(unsigned int)*BUFF_SIZE) );
	cudaCheckError(  __FILE__, __LINE__, cudaMalloc( &d_idx, sizeof(unsigned int)) );
    bfs_kernel_dp_cons_prepare<<<1,1>>>(d_levelArray, d_buffer, d_idx, source);
	
	int children = 1;
#if (CONSOLIDATE_LEVEL==0)
	bfs_kernel_dp_warp_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
												d_buffer, d_buffer, d_idx);
#elif (CONSOLIDATE_LEVEL==1)
	bfs_kernel_dp_block_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
												d_buffer, d_buffer, d_idx);
#elif (CONSOLIDATE_LEVEL==2)
	bfs_kernel_dp_grid_cons<<<children, THREADS_PER_BLOCK>>>(d_vertexArray, d_edgeArray, d_levelArray,
												d_buffer, d_buffer, d_idx);
#endif

	cudaCheckError(  __FILE__, __LINE__, cudaGetLastError());
	cudaCheckError(  __FILE__, __LINE__, cudaDeviceSynchronize());
	
	end_time=gettime_ms(); //end timing execution
	printf("===> GPU #4 - nested parallelism consolidation: computation time = %.2f ms.\n", end_time-start_time);
	//gpu_print<<<1,1>>>(d_idx);
	
	//copy the level array from GPU to CPU;
	start_time = gettime_ms();
	cudaCheckError(  __FILE__, __LINE__, cudaMemcpy( graph.levelArray, d_levelArray, sizeof(unsigned)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime_ms();
	printf("mem copy to CPU time = %.2f ms.\n", end_time-start_time);
}

void BFS_REC_GPU()
{
	prepare_gpu();
	
#if (PROFILE_GPU!=0)
	reset_gpu_statistics<<<1,1>>>();
	cudaDeviceSynchronize();
#endif

	double time = gettime_ms();
	switch (config.solution) {
		case 0:  bfs_flat_gpu();	// 
			break;
		case 1:  bfs_rec_dp_naive_gpu();	//
			break;
		case 2:  bfs_rec_dp_hier_gpu();	//
			break;
		case 3:  bfs_rec_dp_cons_gpu();	//
			break;
		default:
			break;
	}
	fprintf(stderr, "Total execution time = %.2f ms.\n", gettime_ms()-time);

#if (PROFILE_GPU!=0)
	gpu_statistics<<<1,1>>>(1);
	cudaDeviceSynchronize();
#endif

	clean_gpu();
}

