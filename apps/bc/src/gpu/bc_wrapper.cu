#include <stdio.h>
#include <cuda.h>
#include "bc.h"

#define QMAXLENGTH 10240
#define GM_BUFF_SIZE 10240

#ifndef CONSOLIDATE_LEVEL
#define CONSOLIDATE_LEVEL 1
#endif

#include "bc_kernel.cu"

int *d_vertexArray;
int *d_edgeArray;
int *d_levelArray;
int *d_work_queue;
char *d_frontier;
char *d_update;

float *d_bc;
float *d_delta;
int *d_sigma;
char *d_p;

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

unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;
unsigned int nonstop = 0;

double start_time, end_time;

double back_start, back_total;
double forward_start, forward_total;
	
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
		dimBlock.x = 32 * (noNodeTotal/32+1);
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
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_nonstop, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bc, sizeof(float)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_delta, sizeof(float)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_sigma, sizeof(int)*noNodeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_p, sizeof(char)*noNodeTotal*noNodeTotal ) );
	
	end_time = gettime();
	if (VERBOSE)
		fprintf(stderr, "cudaMalloc:\t\t%lf\n",end_time-start_time);

	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_levelArray, graph.levelArray, sizeof(int)*noNodeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_update, graph.update, sizeof(char)*noNodeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_queue_length, &queue_length, sizeof(unsigned int), cudaMemcpyHostToDevice) );

	cudaCheckError( __LINE__, cudaMemset(d_bc, 0, sizeof(float)*noNodeTotal) );
	cudaCheckError( __LINE__, cudaMemset(d_sigma, 0, sizeof(int)*noNodeTotal) );
	cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
	
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
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_nonstop);
	cudaFree(d_bc);
	cudaFree(d_delta);
	cudaFree(d_sigma);
}

void bc_bitmap_gpu()
{	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_frontier, sizeof(char)*noNodeTotal ) );

	for (int i=0; i<noNodeTotal; ++i) {
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}	
		/* forward BFS */
		/* initialize the unordered working set */
		bc_bitmap_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_frontier, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		nonstop = 1;
		int dist = 0;

		while (nonstop) {
			cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)));
			forward_bfs_bitmap_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray,
																d_frontier, d_update, d_sigma, 
																d_p, noNodeTotal, dist );
			gen_bitmap_workset_kernel<<<dimGrid, dimBlock>>>( d_frontier, d_update, d_nonstop, noNodeTotal);
			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		}   
	
		/* backward propagation */
		//if (VERBOSE)
		//  fprintf(stderr, "Backward propagation dist: %d\n", dist);
		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
			backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
																d_sigma, d_delta, noNodeTotal, dist );
			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}   
	}
	cudaFree(d_frontier);
}

void bc_queue_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	for (int i=0; i<noNodeTotal; ++i) {
	    if (i==50) break;
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}
		forward_start = gettime();
		/* forward BFS */
		/* initialize the unordered working set */
		bc_queue_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_work_queue, d_queue_length, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		queue_length = 1;
		int dist = 0;

		while (queue_length) {
			if ( queue_length<=maxDegreeT ) {
				dimGridT.x = 1;
			}
			else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
				dimGridT.x = queue_length / maxDegreeT + 1;
			}
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);
			}

			forward_bfs_thread_queue_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
																	d_p, d_work_queue, d_queue_length, noNodeTotal, dist );
			cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
			gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal );
			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		}   
		cudaCheckError( __LINE__, cudaDeviceSynchronize() );
		//back_start = gettime();
		//forward_total += back_start - forward_start;
		/* backward propagation */
		//if (VERBOSE)
		//  fprintf(stderr, "Backward propagation dist: %d\n", dist);
		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
			backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
																d_sigma, d_delta, noNodeTotal, dist );
			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}   
		//cudaCheckError( __LINE__, cudaDeviceSynchronize() );
		//back_total += gettime() - back_start;
	}
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void bc_dual_queue_gpu()
{
    unsigned int queue_max_length_l = QMAXLENGTH*4/5;
	unsigned int queue_max_length_h = QMAXLENGTH*1/5;
	int queue_length_l;
	int queue_length_h;
	int *d_work_queue_l;
	int *d_work_queue_h;
	unsigned int *d_queue_length_l;
	unsigned int *d_queue_length_h;

	int bp_queue_length_l;
	int bp_queue_length_h;
	int *d_bp_work_queue_l;
	int *d_bp_work_queue_h;
	unsigned int *d_bp_queue_length_l;
	unsigned int *d_bp_queue_length_h;

	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue_l, sizeof(int)*queue_max_length_l ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue_h, sizeof(int)*queue_max_length_h ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length_l, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length_h, sizeof(unsigned int) ) );

	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bp_work_queue_l, sizeof(int)*queue_max_length_l ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bp_work_queue_h, sizeof(int)*queue_max_length_h ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bp_queue_length_l, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_bp_queue_length_h, sizeof(unsigned int) ) );
	
	// prepare dual queue for backward propagation
	cudaMemset(d_update, 1, noNodeTotal*sizeof(char));
	gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(   d_vertexArray, d_update, noNodeTotal,
															d_bp_work_queue_l, d_bp_queue_length_l, queue_max_length_l,
															d_bp_work_queue_h, d_bp_queue_length_h, queue_max_length_h );
	cudaCheckError( __LINE__, cudaMemcpy( &bp_queue_length_l, d_bp_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cudaCheckError( __LINE__, cudaMemcpy( &bp_queue_length_h, d_bp_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	
	for (int i=0; i<noNodeTotal; ++i) {
	    if (i==50) break;
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}	
		/* forward BFS */
		/* initialize the unordered working set */
		bc_queue_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_work_queue_l, d_queue_length_l, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		queue_length = 1;
		queue_length_l = 1;
		queue_length_h = 0;
		int dist = 0;

		while (queue_length) {
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

				forward_bfs_thread_queue_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
																		d_p, d_work_queue_l, d_queue_length_l, noNodeTotal, dist );
			}
			if ( queue_length_h!=0 ) {
				if ( queue_length_h<=MAXDIMGRID ) {
					dimGridB.x = queue_length_h;
				}
				else if ( queue_length_h<=MAXDIMGRID*1024 ) {
					dimGridB.x = MAXDIMGRID;
					dimGridB.y = queue_length_h / MAXDIMGRID + 1;
				}
				else {
					fprintf(stderr, "Too many elements in queue\n");
					exit(0);
				}

				forward_bfs_block_queue_kernel<<<dimGridB, dimBlockB>>>(d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
																		d_p, d_work_queue_h, d_queue_length_h, noNodeTotal, dist );
			}	
			cudaCheckError( __LINE__, cudaGetLastError() );

			cudaCheckError( __LINE__, cudaMemset(d_queue_length_l, 0, sizeof(unsigned int)) );
			cudaCheckError( __LINE__, cudaMemset(d_queue_length_h, 0, sizeof(unsigned int)) );
			
			gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(   d_vertexArray, d_update, noNodeTotal,
																	d_work_queue_l, d_queue_length_l, queue_max_length_l,
																	d_work_queue_h, d_queue_length_h, queue_max_length_h );


			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length_l, d_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length_h, d_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
			
			queue_length = queue_length_l + queue_length_h;
		}   
		/* backward propagation */
		//if (VERBOSE)
		//  fprintf(stderr, "Backward propagation dist: %d\n", dist);
		if ( bp_queue_length_l!=0 ) {
			if ( bp_queue_length_l<=maxDegreeT ) {
				dimGridT.x = 1;
			}
			else if ( bp_queue_length_l<=maxDegreeT*MAXDIMGRID ) {
				dimGridT.x = bp_queue_length_l / maxDegreeT + 1;
			}
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);
			}
		}	
		if ( bp_queue_length_h!=0 ) {
			if ( bp_queue_length_h<=MAXDIMGRID ) {
				dimGridB.x = bp_queue_length_h;
			}
			else if ( bp_queue_length_h<=MAXDIMGRID*1024 ) {
				dimGridB.x = MAXDIMGRID;
				dimGridB.y = bp_queue_length_h / MAXDIMGRID + 1;
			}
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);
			}
		}

		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
			//backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
			//													d_sigma, d_delta, noNodeTotal, dist );
			if ( bp_queue_length_l !=0 )
				backward_prop_thread_queue_kernel<<<dimGridT, dimBlockT>>>(d_vertexArray, d_edgeArray, d_levelArray, d_p,
																	d_sigma, d_delta, noNodeTotal, dist, 
																	d_bp_work_queue_l, d_bp_queue_length_l );
			if ( bp_queue_length_h !=0 )
				backward_prop_block_queue_kernel<<<dimGridB, dimBlockB>>>(d_vertexArray, d_edgeArray, d_levelArray, d_p,
																	d_sigma, d_delta, noNodeTotal, dist,
																	d_bp_work_queue_h, d_bp_queue_length_h );

			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}   
	}
	cudaFree(d_work_queue_l);
	cudaFree(d_queue_length_l);
	cudaFree(d_work_queue_h);
	cudaFree(d_queue_length_h);
	
	cudaFree(d_bp_work_queue_l);
	cudaFree(d_bp_queue_length_l);
	cudaFree(d_bp_work_queue_h);
	cudaFree(d_bp_queue_length_h);
}

void bc_shared_delayed_buffer_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	for (int i=0; i<noNodeTotal; ++i) {
	    if (i==50) break;
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}	
		/* forward BFS */
		/* initialize the unordered working set */
		bc_queue_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_work_queue, d_queue_length, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		queue_length = 1;
		int dist = 0;

		while (queue_length) {
			if ( queue_length<=maxDegreeT ) {
				dimGridT.x = 1;
			}
			else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
				dimGridT.x = queue_length / maxDegreeT + 1;
			}
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);
			}

			forward_bfs_queue_shared_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
																	d_p, d_work_queue, d_queue_length, noNodeTotal, dist );
			cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
			gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal );
			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		}   
		/* backward propagation */
		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
	//		backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
	//															d_sigma, d_delta, noNodeTotal, dist );
			backward_shared_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
														d_sigma, d_delta, noNodeTotal, dist );
			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}   
	}
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void bc_global_delayed_buffer_gpu()
{
	unsigned int buffer_size = 0;
	unsigned int *d_buffer_size;
	int *d_buffer;
	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );
	
	for (int i=0; i<noNodeTotal; ++i) {
	    if (i==50) break;
	//for (int i=0; i<5; ++i) {
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}	
		/* forward BFS */
		/* initialize the unordered working set */
		bc_queue_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_work_queue, d_queue_length, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		queue_length = 1;
		int dist = 0;

		while (queue_length) {
			if ( queue_length<=maxDegreeT ) {
				dimGridT.x = 1;
			}
			else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
				dimGridT.x = queue_length / maxDegreeT + 1;
			}
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);
			}
			cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)) );
			forward_bfs_queue_global_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
															d_p, d_work_queue, d_queue_length, d_buffer, d_buffer_size, noNodeTotal, dist );
			cudaCheckError( __LINE__, cudaMemcpy( &buffer_size, d_buffer_size, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
#ifdef CPU_PROFILE
			fprintf(stderr, "Iteration %d - On CPU buffer size : %d\n", dist, buffer_size);
			//fprintf(stderr, "%d\t%d\n", dist, buffer_size);
#endif
			if ( buffer_size!=0 ) {
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

				forward_bfs_block_queue_kernel<<<dimGridB, dimBlockB>>>(d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
																		d_p, d_buffer, d_buffer_size, noNodeTotal, dist );
			}

			cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
	
			gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal );
			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		}   
		/* backward propagation */
		//if (VERBOSE)
		//  fprintf(stderr, "Backward propagation dist: %d\n", dist);
		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
			//backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
			//													d_sigma, d_delta, noNodeTotal, dist );
			cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)) );
			backward_global_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
																	d_sigma, d_delta, d_buffer, d_buffer_size,
																	noNodeTotal, dist );

			cudaCheckError( __LINE__, cudaMemcpy( &buffer_size, d_buffer_size, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
#ifdef CPU_PROFILE
			fprintf(stderr, "Iteration %d - Backward On CPU buffer size : %d\n", dist, buffer_size);
			//fprintf(stderr, "%d\t%d\n", dist, buffer_size);
#endif
			if ( buffer_size!=0 ) {
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
				backward_prop_block_queue_kernel<<<dimGridB, dimBlockB>>>( d_vertexArray, d_edgeArray, d_levelArray, d_p,
																	d_sigma, d_delta, noNodeTotal, dist, d_buffer, d_buffer_size);
			}

			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}   
	}
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);

}

void bc_np_naive_gpu()
{
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );

	for (int i=0; i<noNodeTotal; ++i) {
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}	
		/* forward BFS */
		/* initialize the unordered working set */
		bc_queue_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_work_queue, d_queue_length, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		queue_length = 1;
		int dist = 0;

		while (queue_length) {
			if ( queue_length<=maxDegreeT ) {
				dimGridT.x = 1;
			}
			else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
				dimGridT.x = queue_length / maxDegreeT + 1;
			}
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);
			}

			forward_bfs_thread_queue_multidp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
																	d_p, d_work_queue, d_queue_length, noNodeTotal, dist );
			cudaCheckError( __LINE__, cudaGetLastError() );
			cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
			gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal );
			cudaCheckError( __LINE__, cudaGetLastError() );
			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		}   
		/* backward propagation */
		//if (VERBOSE)
		//  fprintf(stderr, "Backward propagation dist: %d\n", dist);
		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
			backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
																d_sigma, d_delta, noNodeTotal, dist );
			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}   
	}
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
}

void bc_np_opt_gpu()
{
	int *d_buffer;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	
	for (int i=0; i<noNodeTotal; ++i) {
	    if (i==50) break;
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}	
		/* forward BFS */
		/* initialize the unordered working set */
		bc_queue_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_work_queue, d_queue_length, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		queue_length = 1;
		int dist = 0;

		while (queue_length) {
			if ( queue_length<=maxDegreeT ) {
				dimGridT.x = 1;
			}
			else if ( queue_length<=maxDegreeT*MAXDIMGRID ) {
				dimGridT.x = queue_length / maxDegreeT + 1;
			}
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);
			}

			forward_bfs_thread_queue_singledp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma, 
																	d_p, d_work_queue, d_queue_length, d_buffer, noNodeTotal, dist );
			cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
			gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal );
			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		}   
		/* backward propagation */
		//if (VERBOSE)
		//  fprintf(stderr, "Backward propagation dist: %d\n", dist);
		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
		//	backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
		//														d_sigma, d_delta, noNodeTotal, dist );
			backward_singledp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
																d_sigma, d_delta, d_buffer, noNodeTotal, dist );
			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}   
	}
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
}

void bc_np_consolidate_gpu()
{
	int *d_buffer;
	unsigned int *d_buffer_size;
	unsigned int *d_count;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_work_queue, sizeof(int)*queue_max_length ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_queue_length, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_count, sizeof(unsigned int) ) );

	for (int i=0; i<noNodeTotal; ++i) {
		if (VERBOSE) {
			if (i%1000==0) fprintf(stderr, "Processing node %d...\n", i);
			//fprintf(stderr, "Processing node %d...\n", i);
		}
		/* forward BFS */
		/* initialize the unordered working set */
		bc_queue_init_kernel<<<dimGrid, dimBlock>>>(d_sigma, d_work_queue, d_queue_length, d_levelArray, i, noNodeTotal);
		cudaCheckError( __LINE__, cudaMemset(d_p, 0, sizeof(char)*noNodeTotal*noNodeTotal) );
		queue_length = 1;
		int dist = 0;

		while (queue_length) {
			if ( queue_length<=maxDegreeT ) {
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
			forward_bfs_warp_dp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma,
																d_p, d_work_queue, d_queue_length, d_buffer, noNodeTotal, dist );
#elif (CONSOLIDATE_LEVEL==1)
			forward_bfs_block_dp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma,
																d_p, d_work_queue, d_queue_length, d_buffer, noNodeTotal, dist );
#elif (CONSOLIDATE_LEVEL==2)
			cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)));
			cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
			forward_bfs_grid_dp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_update, d_sigma,
																d_p, d_work_queue, d_queue_length, d_buffer, noNodeTotal, dist,
																d_buffer_size, d_count);
#endif
			cudaCheckError( __LINE__, cudaMemset(d_queue_length, 0, sizeof(unsigned int)) );
			gen_queue_workset_kernel<<<dimGrid, dimBlock>>>( d_update, d_work_queue, d_queue_length, queue_max_length, noNodeTotal );
			dist++;
			cudaCheckError( __LINE__, cudaMemcpy( &queue_length, d_queue_length, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
		}
		/* backward propagation */
		//if (VERBOSE)
		//  fprintf(stderr, "Backward propagation dist: %d\n", dist);
		backward_init_kernel<<<dimGrid, dimBlock>>>(d_delta, d_sigma, noNodeTotal);

		while (dist>1) {
		//	backward_propagation_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
		//														d_sigma, d_delta, noNodeTotal, dist );
#if (CONSOLIDATE_LEVEL==0)
			backward_warp_dp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
															d_sigma, d_delta, d_buffer, noNodeTotal, dist );
#elif (CONSOLIDATE_LEVEL==1)
			backward_block_dp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
																		d_sigma, d_delta, d_buffer, noNodeTotal, dist );
#elif (CONSOLIDATE_LEVEL==2)
			cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)));
			cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
			backward_grid_dp_kernel<<<dimGrid, dimBlock>>>(	d_vertexArray, d_edgeArray, d_levelArray, d_p,
															d_sigma, d_delta, d_buffer, noNodeTotal, dist,
															d_buffer_size, d_count);
#endif
			backward_sum_kernel<<<dimGrid, dimBlock>>>( d_levelArray, d_bc, d_delta, noNodeTotal, dist, i );
			dist--;
		}
	}
	cudaFree(d_work_queue);
	cudaFree(d_queue_length);
	cudaFree(d_buffer);
}

void BC_GPU()
{
	back_total = 0.0;
	forward_total = 0.0;
	prepare_gpu();

	start_time = gettime();
	switch (config.solution) {
		case 0:  bc_bitmap_gpu();	// 
			break;
		case 1:  bc_queue_gpu();	//
			break;
		case 2:  bc_dual_queue_gpu();	//
			break;
		case 3:  bc_shared_delayed_buffer_gpu();	//
			break;
		case 4:  bc_global_delayed_buffer_gpu();	//
			break;
		case 5:  bc_np_naive_gpu();	//
			break;
		case 6:  bc_np_opt_gpu();	//
			break;
		case 7:  bc_np_consolidate_gpu();	//
					break;
		default:
			break;
	}
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	end_time = gettime();
//	fprintf(stdout, "Forward time:\t\t%lf\n", forward_total);
//	fprintf(stdout, "Forward percentage :\t\t%lf\n", forward_total/(end_time-start_time));
//	fprintf(stdout, "Backward time:\t\t%lf\n", back_total);
//	fprintf(stdout, "Backward percentage :\t\t%lf\n", back_total/(end_time-start_time));
	fprintf(stdout, "Execution time:\t\t%lf\n", end_time-start_time);
	start_time = end_time;
	cudaCheckError( __LINE__, cudaMemcpy( bc, d_bc, sizeof(float)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime();
	fprintf(stdout, "cudaMemcpy to CPU:\t\t%lf\n", end_time-start_time);

	if (DEBUG) {
		//cudaCheckError( __LINE__, cudaMemcpy( graph.levelArray, d_levelArray, sizeof(int)*noNodeTotal, cudaMemcpyDeviceToHost) );
		//outputLevel();	
	}
	
	clean_gpu();
}

