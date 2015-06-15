#include <stdio.h>
#include <cuda.h>
#include "graph_color.h"

#define QMAXLENGTH 10240000
#define GM_BUFF_SIZE 10240000

#define WARP_SIZE 32

#ifndef CONSOLIDATE_LEVEL
#define CONSOLIDATE_LEVEL 1
#endif

#include "graph_color_kernel.cu"

int *d_vertexArray;
int *d_edgeArray;
int *d_colorArray;
int *d_work_queue;

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
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_colorArray, sizeof(int)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_nonstop, sizeof(unsigned int) ) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "cudaMalloc:\t\t%lf\n",end_time-start_time);

	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_colorArray, graph.colorArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "cudaMemcpy:\t\t%lf\n", end_time-start_time);
}

void clean_gpu()
{
	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_colorArray);
	cudaFree(d_nonstop);
}

void gclr_nopruning_gpu()
{	
	/* prepare GPU */

	nonstop = 1;
	int color_type = 1;

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)));
		gclr_bitmap_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_colorArray,
												  d_nonstop, color_type, noNodeTotal );
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}   

	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
}

void gclr_queue_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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
		gclr_thread_queue_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_colorArray, color_type,
															noNodeTotal, d_work_queue, d_queue_length);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		if (DEBUG)
			fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void gclr_dual_queue_gpu()
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

	/* initialize the dual working set */
	gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_colorArray, noNodeTotal,
														d_work_queue_l, d_queue_length_l, queue_max_length_l,
														d_work_queue_h, d_queue_length_h, queue_max_length_h );
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_l, d_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_h, d_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	queue_length = queue_length_l + queue_length_h;

	int color_type = 1;

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

			gclr_thread_queue_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_colorArray, color_type,
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
			gclr_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_colorArray, color_type,
																noNodeTotal, d_work_queue_h, d_queue_length_h );
		}
		cudaCheckError( __LINE__, cudaGetLastError() );

		cudaCheckError( __LINE__, cudaMemset(d_queue_length_l, 0, sizeof(unsigned int)) );
		cudaCheckError( __LINE__, cudaMemset(d_queue_length_h, 0, sizeof(unsigned int)) );
		gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_colorArray, noNodeTotal,
																d_work_queue_l, d_queue_length_l, queue_max_length_l,
																d_work_queue_h, d_queue_length_h, queue_max_length_h );
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length_l, d_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length_h, d_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		queue_length = queue_length_l + queue_length_h;

		//if (DEBUG)
		//	fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}

	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue_l);
	cudaFree(d_queue_length_l);
	cudaFree(d_work_queue_h);
	cudaFree(d_queue_length_h);
}

void gclr_shared_delayed_buffer_gpu()
{
	/* prepare GPU */

	nonstop = 1;
	int color_type = 1;

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)));
		gclr_bitmap_shared_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(d_vertexArray, d_edgeArray, d_colorArray,
												  d_nonstop, color_type, noNodeTotal );
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	}   

	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
}

void gclr_global_delayed_buffer_gpu()
{
	/* prepare GPU */
	unsigned int buffer_size = 0;
	unsigned int *d_buffer_size;
	int *d_buffer;
	
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	nonstop = 1;
	int color_type = 1;

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)) );
		gclr_bitmap_global_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_colorArray,
																			d_nonstop, color_type, d_buffer, 
																			d_buffer_size, noNodeTotal );
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
		gclr_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_colorArray, color_type,
															noNodeTotal, d_buffer, d_buffer_size );
            
		color_type++;
		if (buffer_size!=0) 
			nonstop = 1;
		else
			cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		
		//if (DEBUG)
		//	fprintf(stderr, "Iteration: %d  Buffer size: %d\n", color_type-1, buffer_size);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_buffer);
	cudaFree(d_buffer_size);
}

void gclr_np_naive_gpu()
{
	/* prepare GPU */

	/* initialize the unordered working set */
	nonstop = 1;
	int color_type = 1;

	while (nonstop) {
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)));
		gclr_bitmap_multidp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_colorArray, color_type,
															noNodeTotal, d_work_queue, d_queue_length);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

		fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);

}

void gclr_np_opt_gpu()
{
	int *d_buffer;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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

		gclr_queue_singledp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
															noNodeTotal, d_work_queue, d_queue_length, d_buffer);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		if (DEBUG)
			fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
}

void gclr_np_consolidate_gpu()
{
	int *d_buffer;
	unsigned int *d_buf_size;
	unsigned int *d_count;

	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buf_size, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_count, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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
#if (CONSOLIDATE_LEVEL==0)
		gclr_consolidate_warp_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																	noNodeTotal, d_work_queue, d_queue_length, d_buffer);
#elif (CONSOLIDATE_LEVEL==1)
		gclr_consolidate_block_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																noNodeTotal, d_work_queue, d_queue_length, d_buffer);
#elif (CONSOLIDATE_LEVEL==2)
		cudaCheckError( __LINE__, cudaMemset(d_buf_size, 0, sizeof(unsigned int)));
		cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
		gclr_consolidate_grid_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																noNodeTotal, d_work_queue, d_queue_length, d_buffer,
																d_buf_size, d_count);
#endif
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		if (DEBUG)
			fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
}

void gclr_queue_shared_delayed_buffer_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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

		gclr_queue_shared_delayed_buffer_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																		d_work_queue, d_queue_length, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

		//fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void gclr_queue_global_delayed_buffer_gpu()
{

	unsigned int buffer_size = 0;
	unsigned int *d_buffer_size;
	int *d_buffer;
	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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
		cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)));
		gclr_queue_global_delayed_buffer_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_colorArray,
															color_type,	d_work_queue, d_queue_length,
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
		gclr_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_vertexArray, d_edgeArray, d_colorArray, color_type,
															noNodeTotal, d_buffer, d_buffer_size );
            
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		if (DEBUG)
			fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
	cudaFree(d_buffer_size);
}

void gclr_queue_np_naive_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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

		gclr_queue_multidp_kernel<<<dimGridT, dimBlockT>>>(	d_vertexArray, d_edgeArray, d_colorArray, color_type,
															noNodeTotal, d_work_queue, d_queue_length);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

		if (DEBUG)
			fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void gclr_queue_np_opt_gpu()
{
	int *d_buffer;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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

		gclr_queue_singledp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
															noNodeTotal, d_work_queue, d_queue_length, d_buffer);
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		if (DEBUG)
			fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
}

void gclr_queue_np_consolidate_gpu()
{
	int *d_buffer;
	unsigned int *d_buf_size;
	unsigned int *d_count;

	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buf_size, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_count, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	int color_type = 1;
	gen_queue_workset_kernel<<<dimGrid, dimBlock>>>(d_colorArray, d_work_queue, d_queue_length,
													queue_max_length, noNodeTotal);
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

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
#if (CONSOLIDATE_LEVEL==0)
		gclr_consolidate_warp_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																	noNodeTotal, d_work_queue, d_queue_length, d_buffer);
#elif (CONSOLIDATE_LEVEL==1)
		gclr_consolidate_block_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																noNodeTotal, d_work_queue, d_queue_length, d_buffer);
#elif (CONSOLIDATE_LEVEL==2)
		cudaCheckError( __LINE__, cudaMemset(d_buf_size, 0, sizeof(unsigned int)));
		cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
		gclr_consolidate_grid_dp_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_colorArray, color_type,
																noNodeTotal, d_work_queue, d_queue_length, d_buffer,
																d_buf_size, d_count);
#endif
		cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
		gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_colorArray, d_work_queue, d_queue_length, queue_max_length, noNodeTotal);
		color_type++;
		cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		if (DEBUG)
			fprintf(stderr, "Iteration: %d  Queue length: %d\n", color_type-1, queue_length);
	}
	if (DEBUG)
		fprintf(stderr, "Graph Coloring ends in %d iterations.\n", color_type-1);
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
}

void GRAPH_COLOR_GPU()
{
	prepare_gpu();
	
	start_time = gettime();
	switch (config.solution) {
		case 0:  gclr_nopruning_gpu();	//
			break;
		case 1:  gclr_queue_gpu();	//
			break;
		case 2:  gclr_dual_queue_gpu();	//
			break;
		case 3:  gclr_shared_delayed_buffer_gpu();	//
			break;
		case 4:  gclr_global_delayed_buffer_gpu();	//
			break;
		case 5:  gclr_np_naive_gpu();	//
			break;
		case 6:  gclr_np_opt_gpu();	//
			break;
		case 7:  gclr_np_consolidate_gpu();	//
			break;
		case 8:  gclr_queue_shared_delayed_buffer_gpu();	//
			break;
		case 9:  gclr_queue_global_delayed_buffer_gpu();	//
			break;
		case 10:  gclr_queue_np_naive_gpu();	//
			break;
		case 11:  gclr_queue_np_opt_gpu();	//
			break;
		case 12:  gclr_queue_np_consolidate_gpu();	//
			break;
		default:
			break;
	}
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	end_time = gettime();
	fprintf(stderr, "Execution time:\t\t%lf\n", end_time-start_time);
	cudaCheckError( __LINE__, cudaMemcpy( graph.colorArray, d_colorArray, sizeof(int)*noNodeTotal, cudaMemcpyDeviceToHost) );
	clean_gpu();
}

