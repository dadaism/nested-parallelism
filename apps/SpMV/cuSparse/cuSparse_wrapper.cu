#include <stdio.h>
#include <cuda.h>
#include "cusparse.h"
#include "cuSparse.h"

int *d_vertexArray;
int *d_edgeArray;
int *d_levelArray;
int *d_work_queue;
char *d_frontier;
FLOAT_T *d_data;
FLOAT_T *d_x;
FLOAT_T *d_y;

unsigned int *d_queue_length;
unsigned int *d_nonstop;

cusparseStatus_t status;
cusparseHandle_t handle = 0;
cusparseMatDescr_t descr = 0;

inline void cusparseCheckError(int line, cusparseStatus_t ce)
{
	if (ce != CUSPARSE_STATUS_SUCCESS){
		printf("Error: line %d %s\n", line );//cudaGetErrorString(ce));
		exit(1);
	}
}

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
	
	/* Allocate GPU memory */
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_vertexArray, sizeof(int)*(noNodeTotal+1) ) );
	cudaCheckError( __LINE__, cudaMalloc( (void**)&d_edgeArray, sizeof(int)*noEdgeTotal ) );
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
		data[i] = rand() % 10 + 0.5;
		//data[i] = i%13 + 0.5;
	for (int i=0; i<noNodeTotal; ++i) 
		x[i] = rand() % 10 + 0.5;
		//x[i] = i%17 + 0.5;
	
	start_time = gettime();
	cudaCheckError( __LINE__, cudaMemcpy( d_vertexArray, graph.vertexArray, sizeof(int)*(noNodeTotal+1), cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_edgeArray, graph.edgeArray, sizeof(int)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_data, data, sizeof(FLOAT_T)*noEdgeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemcpy( d_x, x, sizeof(FLOAT_T)*noNodeTotal, cudaMemcpyHostToDevice) );
	cudaCheckError( __LINE__, cudaMemset(d_y, 0, sizeof(FLOAT_T)*noNodeTotal) );
	end_time = gettime();
	h2d_memcpy_time += end_time-start_time;

	start_time = gettime();
	/* initialize cusparse library */ 
	cusparseCheckError( __LINE__, cusparseCreate(&handle) ); 

	/* create and setup matrix descriptor */ 
	cusparseCheckError( __LINE__, cusparseCreateMatDescr(&descr) ); 

	cusparseCheckError( __LINE__, cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL) ); 
	cusparseCheckError( __LINE__, cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO) ); 

	end_time = gettime();
	cusparse_init_time += end_time-start_time;
}

int clean_gpu()
{
	cudaCheckError( __LINE__, cudaFree(d_vertexArray) );
	cudaCheckError( __LINE__, cudaFree(d_edgeArray) );
	cudaCheckError( __LINE__, cudaFree(d_data) );
	cudaCheckError( __LINE__, cudaFree(d_x) );
	cudaCheckError( __LINE__, cudaFree(d_y) );

	/* destroy matrix descriptor */ 
	cusparseCheckError( __LINE__, cusparseDestroyMatDescr(descr) ); 
	descr = 0; 

	/* destroy handle */ 
	cusparseCheckError( __LINE__, cusparseDestroy(handle) ); 
	handle = 0; 
	
	return 0;
}

void cusparse_gpu()
{
	FLOAT_T alpha = 1;
	FLOAT_T beta = 0;
	
	//csrRowPtr = [0 3 4 7 9] 
	/* exercise Level 2 routines (csrmv) */ 
	status= cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, noNodeTotal, noNodeTotal, noEdgeTotal, 
						   &alpha, descr, d_data, d_vertexArray, d_edgeArray, 
						   d_x, &beta, d_y); 

	if (status != CUSPARSE_STATUS_SUCCESS) { 
		printf("Matrix-vector multiplication failed"); 
	} 
}

void cuSparse()
{
	prepare_gpu();
	
	start_time = gettime();
	cusparse_gpu();
	cudaCheckError( __LINE__, cudaDeviceSynchronize() );
	end_time = gettime();
	ker_exe_time += end_time - start_time;

	start_time = end_time;
	cudaCheckError( __LINE__, cudaMemcpy( y, d_y, sizeof(FLOAT_T)*noNodeTotal, cudaMemcpyDeviceToHost) );
	end_time = gettime();

    d2h_memcpy_time += end_time-start_time;
	clean_gpu();
}

