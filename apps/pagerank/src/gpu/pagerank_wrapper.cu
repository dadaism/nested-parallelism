#include <stdio.h>
#include <cuda.h>
#include "pagerank.h"

#define QMAXLENGTH 10240000
#define GM_BUFF_SIZE 10240000
#define DAMPING_FACTOR 0.5
#define EPSILON 1.0e-13

#define THREADS_PER_BLOCK 192

#ifndef CONSOLIDATE_LEVEL
#define CONSOLDIATE_LEVEL 1
#endif

#include "pagerank_kernel.cu"

//#define CPU_PROFILE

int *d_levelArray;

int *d_vertexArray;
int *d_edgeArray;
int *d_outdegreeArray;
int *d_childVertexArray;
int *d_rEdgeArray;
int *d_work_queue;
char *d_frontier;
char *d_update;
int *d_noDanglingNode;
int *d_danglingVertexArray;
FLOAT_T *d_danglingRankArray;
FLOAT_T *d_rankArray;
FLOAT_T *d_newRankArray;
FLOAT_T *d_temp;

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
dim3 dimGridD(1,1,1);   // thread mapping for dangling array
dim3 dimBlockD(1,1,1);

//char *update = new char [noNodeTotal] ();
//int *queue = new int [queue_max_length];
unsigned int queue_max_length = QMAXLENGTH;
unsigned int queue_length = 0;
unsigned int nonstop = 0;

// generate queue for dangling nodes
int noDanglingNode = 0;
int *danglingVertexArray;
FLOAT_T *danglingRankArray;
 
FLOAT_T dangling_rank = 0.0;
FLOAT_T old_dangling_rank = 0.0;
FLOAT_T damping = DAMPING_FACTOR;
FLOAT_T rank_random_walk;
FLOAT_T rank_dangling_node;
FLOAT_T delta_rank_dangling_node;

double start_time, end_time;
double dangling_start;

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
		//fprintf(stderr, "cudaSetDevice:\t\t%lf\n",end_time-start_time);
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
	
	memset( graph.rankArray, 0, sizeof(FLOAT_T)*noNodeTotal );
	
	/* Allocate GPU memory */
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_outdegreeArray, sizeof(int)*noNodeTotal ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_childVertexArray, sizeof(int)*(noNodeTotal+1) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_rEdgeArray, sizeof(int)*noEdgeTotal ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_rankArray, sizeof(FLOAT_T)*noNodeTotal ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_newRankArray, sizeof(FLOAT_T)*noNodeTotal ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_update, sizeof(char)*noNodeTotal ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_noDanglingNode, sizeof(int) ) );
    cudaCheckError( __LINE__, cudaMalloc( (void**)&d_nonstop, sizeof(unsigned int) ) );

	end_time = gettime();
	d_malloc_time += end_time - start_time;

	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
    cudaCheckError( __LINE__, cudaMemcpy( d_outdegreeArray, graph.outdegreeArray, sizeof(int)*(noNodeTotal), cudaMemcpyHostToDevice) );
    cudaCheckError( __LINE__, cudaMemcpy( d_childVertexArray, graph.childVertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
    cudaCheckError( __LINE__, cudaMemcpy( d_rEdgeArray, graph.rEdgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
    cudaCheckError( __LINE__, cudaMemcpy( d_rankArray, graph.rankArray, sizeof(FLOAT_T)*noNodeTotal, cudaMemcpyHostToDevice) );
    cudaCheckError( __LINE__, cudaMemset( d_newRankArray, 0, sizeof(FLOAT_T)*noNodeTotal) );
    //cudaCheckError( __LINE__, cudaMemset( d_update, 1, sizeof(char)*noNodeTotal) );	//	BE CAREFULL!! d_update should be set to 1
	
	end_time = gettime();
	h2d_memcpy_time += end_time - start_time;
}

void prepare_dangling_node()
{
	noDanglingNode = 0;
	// generate queue for dangling nodes
	for (int i=0; i<noNodeTotal; ++i) {
		int start = graph.vertexArray[i];
		int end = graph.vertexArray[i+1];
		if ( start==end )
			noDanglingNode++;
    }   
    if ( noDanglingNode!=0 ) {
        danglingVertexArray = new int [noDanglingNode] (); 
        danglingRankArray = new FLOAT_T [noDanglingNode] ();
        int index = 0;
        for (int i=0; i<noNodeTotal; ++i) {
            int start = graph.vertexArray[ i ];
            int end = graph.vertexArray[ i+1 ];
            if ( start==end )
                danglingVertexArray[index++] = i;
		}   
		cudaCheckError( __LINE__, cudaMalloc( (void**)&d_danglingVertexArray, sizeof(int)*noDanglingNode ) );
		cudaCheckError( __LINE__, cudaMalloc( (void**)&d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode ) );
		cudaCheckError( __LINE__, cudaMemcpy( d_noDanglingNode, &noDanglingNode, sizeof(int), cudaMemcpyHostToDevice) );
		cudaCheckError( __LINE__, cudaMemcpy( d_danglingVertexArray, danglingVertexArray, sizeof(int)*noDanglingNode, cudaMemcpyHostToDevice) );
	
		/* Configuration for dangline thread mapping for all nodes */    
		if ( noDanglingNode > maxDegreeT ){
			dimGridD.x = noDanglingNode / maxDegreeT + 1;
			dimBlockD.x = maxDegreeT;
		}
		else {
			dimGridD.x = 1;
			dimBlockD.x = noDanglingNode;
		}
	}

	rank_random_walk = (1-damping)*TOTAL_RANK/noNodeTotal;
}

void clean_gpu()
{
	cudaFree(d_vertexArray);
	cudaFree(d_edgeArray);
	cudaFree(d_outdegreeArray);
	cudaFree(d_childVertexArray);
	cudaFree(d_rEdgeArray);
	cudaFree(d_rankArray);
	cudaFree(d_newRankArray);
	cudaFree(d_update);
	cudaFree(d_noDanglingNode);
}

void pg_pull_gpu()
{
	/* prepare GPU */


	/* initialize the unordered working set */
	nonstop = 1;
	int iteration = 0;
	while (nonstop) {
		dangling_start = gettime();
		// collecting danglingRank for this iteration
		old_dangling_rank = dangling_rank;
		dangling_rank = 0.0; // accumulated dangling_rank
		if ( noDanglingNode!=0 ) {
			update_danglingrankarray_kernel<<<dimGridD, dimBlockD>>>(d_rankArray, d_danglingRankArray, d_danglingVertexArray, noDanglingNode);
            cudaCheckError( __LINE__, cudaMemcpy(danglingRankArray, d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode, cudaMemcpyDeviceToHost) );
            for (int i=0; i<noDanglingNode; ++i) {
                dangling_rank += danglingRankArray[i];
            }
        }
		dangling_time += gettime() - dangling_start;
		
        rank_dangling_node = damping * dangling_rank / noNodeTotal;
        delta_rank_dangling_node = damping * (dangling_rank - old_dangling_rank)/noNodeTotal;
	
		/* pull method */
		pagerank_thread_pull_kernel<<<dimGrid, dimBlock>>>(d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
															d_rankArray, d_newRankArray, rank_random_walk, 
															rank_dangling_node, damping, noNodeTotal );
        
		cudaCheckError( __LINE__, cudaGetLastError() );
		
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		check_delta_kernel<<<dimGrid, dimBlock>>>( d_rankArray, d_newRankArray, d_nonstop, noNodeTotal);
        cudaCheckError( __LINE__, cudaMemcpy(&nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
       
		/* switch pointers to avoid unnecessary copy */
		d_temp = d_rankArray;
		d_rankArray = d_newRankArray;
		d_newRankArray = d_temp;

		iteration++;
	}
	if (DEBUG)
		fprintf(stderr, "PageRank ends in %d iterations.\n", iteration); 
}

void pg_dual_queue_gpu()
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
	cudaCheckError( __LINE__, cudaMemset(d_update, 1, sizeof(char)*noNodeTotal) );
	cudaCheckError( __LINE__, cudaMemset(d_queue_length_l, 0, sizeof(unsigned int)) );
	cudaCheckError( __LINE__, cudaMemset(d_queue_length_h, 0, sizeof(unsigned int)) );
	gen_dual_queue_workset_kernel<<<dimGrid, dimBlock>>>(	d_childVertexArray, d_update, noNodeTotal,
															d_work_queue_l, d_queue_length_l, queue_max_length_l,
															d_work_queue_h, d_queue_length_h, queue_max_length_h );
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_l, d_queue_length_l, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cudaCheckError( __LINE__, cudaMemcpy( &queue_length_h, d_queue_length_h, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	if (DEBUG) {
		fprintf(stderr, "Queue size of low outdegree:\t%d\n", queue_length_l);
		fprintf(stderr, "Queue size of high outdegree:\t%d\n", queue_length_h);
	}

	nonstop = 1;
	int iteration = 0;	
	while (nonstop) {
		// collecting danglingRank for this iteration
		old_dangling_rank = dangling_rank;
		dangling_rank = 0.0; // accumulated dangling_rank
		if ( noDanglingNode!=0 ) {
			update_danglingrankarray_kernel<<<dimGridD, dimBlockD>>>(d_rankArray, d_danglingRankArray, d_danglingVertexArray, noDanglingNode);
            cudaCheckError( __LINE__, cudaMemcpy(danglingRankArray, d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode, cudaMemcpyDeviceToHost) );
            for (int i=0; i<noDanglingNode; ++i) {
                dangling_rank += danglingRankArray[i];
            }
		}
		
        rank_dangling_node = damping * dangling_rank / noNodeTotal;
        //delta_rank_dangling_node = damping * (dangling_rank - old_dangling_rank)/noNodeTotal;
		
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
			pg_thread_queue_kernel<<<dimGridT, dimBlockT>>>(	d_childVertexArray, d_rEdgeArray, d_outdegreeArray, d_rankArray,
																d_newRankArray, rank_random_walk, rank_dangling_node, damping,
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
			pg_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_childVertexArray, d_rEdgeArray, d_outdegreeArray, d_rankArray,
															d_newRankArray, rank_random_walk, rank_dangling_node, damping,
															noNodeTotal, d_work_queue_h, d_queue_length_h );
		}
		cudaCheckError( __LINE__, cudaGetLastError() );

		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		check_delta_kernel<<<dimGrid, dimBlock>>>( d_rankArray, d_newRankArray, d_nonstop, noNodeTotal);
        cudaCheckError( __LINE__, cudaMemcpy(&nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
    
		/* switch pointers to avoid unnecessary copy */
		d_temp = d_rankArray;
		d_rankArray = d_newRankArray;
		d_newRankArray = d_temp;
		
		iteration++;
	}
	if (DEBUG)
		fprintf(stderr, "PageRank ends in %d iterations.\n", iteration); 
	cudaFree(d_work_queue_l);
	cudaFree(d_queue_length_l);
	cudaFree(d_work_queue_h);
	cudaFree(d_queue_length_h);
}

void pg_shared_delayed_buffer_gpu()
{
	/* prepare GPU */

	/* initialize the unordered working set */
	nonstop = 1;

	int iteration = 0;
	while (nonstop) {
		// collecting danglingRank for this iteration
		old_dangling_rank = dangling_rank;
		dangling_rank = 0.0; // accumulated dangling_rank
		if ( noDanglingNode!=0 ) {
			update_danglingrankarray_kernel<<<dimGridD, dimBlockD>>>(d_rankArray, d_danglingRankArray, d_danglingVertexArray, noDanglingNode);
            cudaCheckError( __LINE__, cudaMemcpy(danglingRankArray, d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode, cudaMemcpyDeviceToHost) );
            for (int i=0; i<noDanglingNode; ++i) {
                dangling_rank += danglingRankArray[i];
            }
		}
		
        rank_dangling_node = damping * dangling_rank / noNodeTotal;
        delta_rank_dangling_node = damping * (dangling_rank - old_dangling_rank)/noNodeTotal;
	
		/* pull method */
		pg_shared_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
																d_rankArray, d_newRankArray, rank_random_walk, 
																rank_dangling_node, damping, noNodeTotal );
        
		cudaCheckError( __LINE__, cudaGetLastError() );
		
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		check_delta_kernel<<<dimGrid, dimBlock>>>( d_rankArray, d_newRankArray, d_nonstop, noNodeTotal);
        cudaCheckError( __LINE__, cudaMemcpy(&nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
       
		/* switch pointers to avoid unnecessary copy */
		d_temp = d_rankArray;
		d_rankArray = d_newRankArray;
		d_newRankArray = d_temp;

		iteration++;
	}
	if (DEBUG)
		fprintf(stderr, "PageRank ends in %d iterations.\n", iteration); 
}

void pg_global_delayed_buffer_gpu()
{

	unsigned int buffer_size = 0;
	unsigned int *d_buffer_size;
	int *d_buffer;
	
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );

	/* initialize the unordered working set */
	nonstop = 1;

	int iteration = 0;
	while (nonstop) {
		// collecting danglingRank for this iteration
		old_dangling_rank = dangling_rank;
		dangling_rank = 0.0; // accumulated dangling_rank
		if ( noDanglingNode!=0 ) {
			update_danglingrankarray_kernel<<<dimGridD, dimBlockD>>>(d_rankArray, d_danglingRankArray, d_danglingVertexArray, noDanglingNode);
            cudaCheckError( __LINE__, cudaMemcpy(danglingRankArray, d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode, cudaMemcpyDeviceToHost) );
            for (int i=0; i<noDanglingNode; ++i) {
                dangling_rank += danglingRankArray[i];
            }
		}
		
        rank_dangling_node = damping * dangling_rank / noNodeTotal;
        delta_rank_dangling_node = damping * (dangling_rank - old_dangling_rank)/noNodeTotal;
	
		/* pull method */
		cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)) );
		pg_global_delayed_buffer_kernel<<<dimGrid, dimBlock>>>(	d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
																d_rankArray, d_newRankArray, rank_random_walk, 
																rank_dangling_node, damping, noNodeTotal, d_buffer,
																d_buffer_size );
        
		cudaCheckError( __LINE__, cudaGetLastError() );
		cudaCheckError( __LINE__, cudaMemcpy(&buffer_size, d_buffer_size, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
#ifdef CPU_PROFILE
		fprintf(stderr, "Iteration %d - On CPU buffer size : %d\n", iteration, buffer_size);
		//fprintf(stderr, "%d\t%d\n", iteration, buffer_size);
#endif
		if ( buffer_size!=0 ) {
			if (buffer_size<=MAXDIMGRID) {
				dimGridB.x = buffer_size;                                                                     
			}
			else if (buffer_size<=MAXDIMGRID*1024) {                               
				dimGridB.x = MAXDIMGRID;
				dimGridB.y = buffer_size/MAXDIMGRID+1;
			}                   
			else {
				fprintf(stderr, "Too many elements in queue\n");
				exit(0);        
			}
		
			pg_block_queue_kernel<<<dimGridB, dimBlockB>>>(	d_childVertexArray, d_rEdgeArray, d_outdegreeArray, d_rankArray,
															d_newRankArray, rank_random_walk, rank_dangling_node, damping, 
															noNodeTotal, d_buffer, d_buffer_size );
            
		}
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		check_delta_kernel<<<dimGrid, dimBlock>>>( d_rankArray, d_newRankArray, d_nonstop, noNodeTotal);
        cudaCheckError( __LINE__, cudaMemcpy(&nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
       
		/* switch pointers to avoid unnecessary copy */
		d_temp = d_rankArray;
		d_rankArray = d_newRankArray;
		d_newRankArray = d_temp;

		iteration++;
	}
	if (DEBUG)
		fprintf(stderr, "PageRank ends in %d iterations.\n", iteration); 

	cudaFree(d_buffer);
	cudaFree(d_buffer_size);
}

void pg_np_naive_gpu()
{
	/* prepare GPU */

	/* initialize the unordered working set */
	nonstop = 1;

	int iteration = 0;
	while (nonstop) {
		// collecting danglingRank for this iteration
		old_dangling_rank = dangling_rank;
		dangling_rank = 0.0; // accumulated dangling_rank
		if ( noDanglingNode!=0 ) {
			update_danglingrankarray_kernel<<<dimGridD, dimBlockD>>>(d_rankArray, d_danglingRankArray, d_danglingVertexArray, noDanglingNode);
            cudaCheckError( __LINE__, cudaMemcpy(danglingRankArray, d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode, cudaMemcpyDeviceToHost) );
            for (int i=0; i<noDanglingNode; ++i) {
                dangling_rank += danglingRankArray[i];
            }
		}
		
        rank_dangling_node = damping * dangling_rank / noNodeTotal;
        delta_rank_dangling_node = damping * (dangling_rank - old_dangling_rank)/noNodeTotal;
	
		/* naive dynamic parallelism method */

		pg_multidp_kernel<<<dimGrid, dimBlock>>>(	d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
													d_rankArray, d_newRankArray, rank_random_walk, 
													rank_dangling_node, damping, noNodeTotal );
        
		cudaCheckError( __LINE__, cudaGetLastError() );
		
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		check_delta_kernel<<<dimGrid, dimBlock>>>( d_rankArray, d_newRankArray, d_nonstop, noNodeTotal);
        cudaCheckError( __LINE__, cudaMemcpy(&nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
       
		/* switch pointers to avoid unnecessary copy */
		d_temp = d_rankArray;
		d_rankArray = d_newRankArray;
		d_newRankArray = d_temp;

		iteration++;
		if (iteration==37) break; // there is a precision bug in this implementation
		// 37 is for CiteSeer only
	}
	if (DEBUG)
		fprintf(stderr, "PageRank ends in %d iterations.\n", iteration); 

}

void pg_np_opt_gpu()
{
	int *d_buffer;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	/* initialize the unordered working set */
	nonstop = 1;

	int iteration = 0;
	while (nonstop) {
		// collecting danglingRank for this iteration
		old_dangling_rank = dangling_rank;
		dangling_rank = 0.0; // accumulated dangling_rank
		if ( noDanglingNode!=0 ) {
			update_danglingrankarray_kernel<<<dimGridD, dimBlockD>>>(d_rankArray, d_danglingRankArray, d_danglingVertexArray, noDanglingNode);
            cudaCheckError( __LINE__, cudaMemcpy(danglingRankArray, d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode, cudaMemcpyDeviceToHost) );
            for (int i=0; i<noDanglingNode; ++i) {
                dangling_rank += danglingRankArray[i];
            }
		}
		
        rank_dangling_node = damping * dangling_rank / noNodeTotal;
        delta_rank_dangling_node = damping * (dangling_rank - old_dangling_rank)/noNodeTotal;
	
		/* pull method */
		pg_singledp_kernel<<<dimGrid, dimBlock>>>(d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
													d_rankArray, d_newRankArray, rank_random_walk, 
													rank_dangling_node, damping, noNodeTotal, d_buffer );
        
		cudaCheckError( __LINE__, cudaGetLastError() );
		
		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		check_delta_kernel<<<dimGrid, dimBlock>>>( d_rankArray, d_newRankArray, d_nonstop, noNodeTotal);
        cudaCheckError( __LINE__, cudaMemcpy(&nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
       
		/* switch pointers to avoid unnecessary copy */
		d_temp = d_rankArray;
		d_rankArray = d_newRankArray;
		d_newRankArray = d_temp;

		iteration++;
	}
	if (DEBUG)
		fprintf(stderr, "PageRank ends in %d iterations.\n", iteration); 

	cudaFree(d_buffer);
}

void pg_np_consolidate_gpu()
{
	int *d_buffer;
	unsigned int *d_buffer_size;
	unsigned int *d_count;
	/* prepare GPU */
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer, sizeof(int)*GM_BUFF_SIZE ) );

	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_buffer_size, sizeof(unsigned int) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_count, sizeof(unsigned int) ) );
	/* initialize the unordered working set */
	nonstop = 1;

	int iteration = 0;
	while (nonstop) {
		// collecting danglingRank for this iteration
		old_dangling_rank = dangling_rank;
		dangling_rank = 0.0; // accumulated dangling_rank
		if ( noDanglingNode!=0 ) {
			update_danglingrankarray_kernel<<<dimGridD, dimBlockD>>>(d_rankArray, d_danglingRankArray, d_danglingVertexArray, noDanglingNode);
            cudaCheckError( __LINE__, cudaMemcpy(danglingRankArray, d_danglingRankArray, sizeof(FLOAT_T)*noDanglingNode, cudaMemcpyDeviceToHost) );
            for (int i=0; i<noDanglingNode; ++i) {
                dangling_rank += danglingRankArray[i];
            }
		}

        rank_dangling_node = damping * dangling_rank / noNodeTotal;
        delta_rank_dangling_node = damping * (dangling_rank - old_dangling_rank)/noNodeTotal;

		/* pull method */
#if (CONSOLIDATE_LEVEL==0)
		pg_warp_dp_kernel<<<dimGrid, dimBlock>>>(d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
												d_rankArray, d_newRankArray, rank_random_walk,
												rank_dangling_node, damping, noNodeTotal, d_buffer );
#elif (CONSOLIDATE_LEVEL==1)
		pg_block_dp_kernel<<<dimGrid, dimBlock>>>(d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
												d_rankArray, d_newRankArray, rank_random_walk,
												rank_dangling_node, damping, noNodeTotal, d_buffer );
#elif (CONSOLIDATE_LEVEL==2)

		cudaCheckError( __LINE__, cudaMemset(d_buffer_size, 0, sizeof(unsigned int)));
		cudaCheckError( __LINE__, cudaMemset(d_count, 0, sizeof(unsigned int)));
		pg_grid_dp_kernel<<<dimGrid, dimBlock>>>(d_childVertexArray, d_rEdgeArray, d_outdegreeArray,
												d_rankArray, d_newRankArray, rank_random_walk,
												rank_dangling_node, damping, noNodeTotal, d_buffer,
												d_buffer_size,d_count );
#endif
		cudaCheckError( __LINE__, cudaGetLastError() );

		cudaCheckError( __LINE__, cudaMemset(d_nonstop, 0, sizeof(unsigned int)) );
		check_delta_kernel<<<dimGrid, dimBlock>>>( d_rankArray, d_newRankArray, d_nonstop, noNodeTotal);
        cudaCheckError( __LINE__, cudaMemcpy(&nonstop, d_nonstop, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

		/* switch pointers to avoid unnecessary copy */
		d_temp = d_rankArray;
		d_rankArray = d_newRankArray;
		d_newRankArray = d_temp;

		iteration++;
	}
	if (DEBUG)
		fprintf(stderr, "PageRank ends in %d iterations.\n", iteration);

	cudaFree(d_buffer);
}

void PAGERANK_GPU()
{
	prepare_gpu();
	prepare_dangling_node();

	start_time = gettime();
	switch (config.solution) {
		case 0:  pg_pull_gpu();	// 
			break;
		case 1:  pg_dual_queue_gpu();	//
			break;
		case 2:  pg_shared_delayed_buffer_gpu();	//
			break;
		case 3:  pg_global_delayed_buffer_gpu();	//
			break;
		case 4:  pg_np_naive_gpu();	//
			break;
		case 5:  pg_np_opt_gpu();	//
			break;
		case 6:  pg_np_consolidate_gpu();	//
					break;
		default:
			break;
	}
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
    
	end_time = gettime();
	ker_exe_time += end_time - start_time;
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( graph.rankArray, d_rankArray, sizeof(FLOAT_T)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime();
	d2h_memcpy_time += end_time - start_time;

	clean_gpu();
}

