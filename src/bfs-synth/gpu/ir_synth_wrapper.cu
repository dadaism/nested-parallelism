#include <stdio.h>
#include <cuda.h>
#include "ir_synth.h"

#define QMAXLENGTH 10240000
#define GM_BUFF_SIZE 10240000

#include "ir_synth_kernel.cu"

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
	if (VERBOSE) {
		fprintf(stderr, "CUDA runtime initialization:\t\t%lf\n",end_time-start_time);
	}
	start_time = gettime();
	cudaCheckError( __LINE__, cudaSetDevice(config.device_num) );
	end_time = gettime();
	if (VERBOSE) {
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
		dimBlock.x = noNodeTotal;
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
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_levelArray, sizeof(int)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_update, sizeof(char)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_nonstop, sizeof(unsigned int) ) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "cudaMalloc:\t\t%lf\n",end_time-start_time);

	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_update, graph.update, sizeof(char)*noNodeTotal, cudaMemcpyHostToDevice) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "cudaMemcpy:\t\t%lf\n", end_time-start_time);
}

void clean_gpu()
{
	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_levelArray);
	cudaFree(d_update);
	cudaFree(d_nonstop);
}

void ir_bitmap_gpu()
{	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_frontier, sizeof(char)*noNodeTotal ) );

	/* forward BFS */
	/* initialize the unordered working set */
	bfs_bitmap_init_kernel<<<dimGrid, dimBlock>>>(d_frontier, d_levelArray, 0, noNodeTotal);
	nonstop = 1;
	int dist = 0;

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)));
		bfs_bitmap_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray,
													d_frontier, d_update, noNodeTotal );
		gen_bitmap_workset_kernel<<<dimGrid, dimBlock>>>( d_frontier, d_update, d_nonstop, noNodeTotal);
		dist++;
		cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}   

	cudaFree(d_frontier);
}

void ir_queue_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	/* forward BFS */
	/* initialize the unordered working set */
	bfs_queue_init_kernel<<<dimGrid, dimBlock>>>(d_levelArray, d_work_queue, d_queue_length, 0, noNodeTotal);
	queue_length = 1;
	int dist = 0;

	while (queue_length) {
		if ( queue_length<=maxDegreeT )	{
			dimGridT.x = 1;
		}
		else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
			dimGridT.x = queue_length / maxDegreeT + 1;
		}
		else {
			fprintf(stderr, "Too many elements in queue\n");
			exit(0);
		}

		bfs_thread_queue_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update,
															noNodeTotal, d_work_queue, d_queue_length);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		dist++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}
	if (DEBUG)
		fprintf(stderr, "BFS ends in %d iterations.\n", dist); 
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void ir_dual_queue_gpu()
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

	/* forward BFS */
	/* initialize the unordered working set */
	bfs_queue_init_kernel<<<dimGrid, dimBlock>>>(d_levelArray, d_work_queue_l, d_queue_length_l, 0, noNodeTotal);
	queue_length = 1;
	queue_length_l = 1;
	queue_length_h = 0;
	int dist = 0;

	while (queue_length) {
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

			bfs_thread_queue_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update,
																noNodeTotal, d_work_queue_l, d_queue_length_l );
		}
		if ( queue_length_h!=0 ) {
			if ( queue_length_h<=MAXDIMGRID ) {
				dimGridB.x = queue_length_h;
			}
			else if ( queue_length_h<=MAXDIMGRID*1024 ) {
				dimGridB.x = MAXDIMGRID;
				dimGridB.y = queue_length_h/MAXDIMGRID+1;
			}
			bfs_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update,
																noNodeTotal, d_work_queue_h, d_queue_length_h );
		
		}
		cudaCheckError( __LINE__, cudaGetLastError() );

		cudaCheckError( __LINE__, cudaMemset(d_queue_length_l, 0, sizeof(unsigned int)) );
		cudaCheckError( __LINE__, cudaMemset(d_queue_length_h, 0, sizeof(unsigned int)) );
		gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_update, noNodeTotal,
																d_work_queue_l, d_queue_length_l, queue_max_length_l,
																d_work_queue_h, d_queue_length_h, queue_max_length_h );
		dist++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length_l, d_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length_h, d_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		
		queue_length = queue_length_l + queue_length_h;
	}

	if (DEBUG)
		fprintf(stderr, "BFS ends in %d iterations.\n", dist); 
	cudaFree(d_work_queue_l);
	cudaFree(d_queue_length_l);
	cudaFree(d_work_queue_h);
	cudaFree(d_queue_length_h);
}

void ir_shared_delayed_buffer_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	/* forward BFS */
	/* initialize the unordered working set */
	bfs_queue_init_kernel<<<dimGrid, dimBlock>>>(d_levelArray, d_work_queue, d_queue_length, 0, noNodeTotal);
	queue_length = 1;
	int dist = 0;

	while (queue_length) {
		if ( queue_length<=maxDegreeT )	{
			dimGridT.x = 1;
		}
		else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
			dimGridT.x = queue_length / maxDegreeT + 1;
		}
		else {
			fprintf(stderr, "Too many elements in queue\n");
			exit(0);
		}

		bfs_queue_shared_delayed_buffer_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_levelArray,
																	d_work_queue, d_queue_length, d_update, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		dist++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}
	if (DEBUG)
		fprintf(stderr, "BFS ends in %d iterations.\n", dist); 
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void ir_global_delayed_buffer_gpu()
{

	unsigned int buffer_size = 0;
	unsigned int *d_buffer_size;
	int *d_buffer;
	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );

	/* forward BFS */
	/* initialize the unordered working set */
	bfs_queue_init_kernel<<<dimGrid, dimBlock>>>(d_levelArray, d_work_queue, d_queue_length, 0, noNodeTotal);
	queue_length = 1;
	int dist = 0;

	while (queue_length) {
		if ( queue_length<=maxDegreeT )	{
			dimGridT.x = 1;
		}
		else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
			dimGridT.x = queue_length / maxDegreeT + 1;
		}
		else {
			fprintf(stderr, "Too many elements in queue\n");
			exit(0);
		}

		bfs_queue_global_delayed_buffer_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_levelArray,
													d_work_queue, d_queue_length, d_update, 
													d_buffer, d_buffer_size, noNodeTotal);
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
		
		bfs_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update,
															noNodeTotal, d_buffer, d_buffer_size );
            
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		dist++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}
	if (DEBUG)
		fprintf(stderr, "BFS ends in %d iterations.\n", dist); 
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
	cudaFree(d_buffer_size);
}

void ir_np_naive_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	/* forward BFS */
	/* initialize the unordered working set */
	bfs_queue_init_kernel<<<dimGrid, dimBlock>>>(d_levelArray, d_work_queue, d_queue_length, 0, noNodeTotal);
	queue_length = 1;
	int dist = 0;

	while (queue_length) {
		if ( queue_length<=maxDegreeT )	{
			dimGridT.x = 1;
		}
		else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
			dimGridT.x = queue_length / maxDegreeT + 1;
		}
		else {
			fprintf(stderr, "Too many elements in queue\n");
			exit(0);
		}

		bfs_thread_queue_multidp_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update,
															noNodeTotal, d_work_queue, d_queue_length);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		dist++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}
	if (DEBUG)
		fprintf(stderr, "BFS ends in %d iterations.\n", dist); 
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void ir_np_opt_gpu()
{
	int *d_buffer;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	/* forward BFS */
	/* initialize the unordered working set */
	bfs_queue_init_kernel<<<dimGrid, dimBlock>>>(d_levelArray, d_work_queue, d_queue_length, 0, noNodeTotal);
	queue_length = 1;
	int dist = 0;

	while (queue_length) {
		if ( queue_length<=maxDegreeT )	{
			dimGridT.x = 1;
		}
		else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
			dimGridT.x = queue_length / maxDegreeT + 1;
		}
		else {
			fprintf(stderr, "Too many elements in queue\n");
			exit(0);
		}

		bfs_thread_queue_singledp_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update,
																	noNodeTotal, d_work_queue, d_queue_length, d_buffer);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		dist++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}
	if (DEBUG)
		fprintf(stderr, "BFS ends in %d iterations.\n", dist); 
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
}

void IR_SYNTH_GPU()
{

	prepare_gpu();
	
	start_time = gettime();
	switch (config.solution) {
		case 0:  ir_bitmap_gpu();	// 
			break;
		case 1:  ir_queue_gpu();	//
			break;
		case 2:  ir_dual_queue_gpu();	//
			break;
		case 3:  ir_shared_delayed_buffer_gpu();	//
			break;
		case 4:  ir_global_delayed_buffer_gpu();	//
			break;
		case 5:  ir_np_naive_gpu();	//
			break;
		case 6:  ir_np_opt_gpu();	//
			break;
		default:
			break;
	}
	end_time = gettime();
	fprintf(stderr, "Execution time:\t\t%lf\n", end_time-start_time);
	clean_gpu();
}

