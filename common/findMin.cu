#include "findMin.h"
#include "findMin_kernel.cu"

int findMin(int *costArray_GPU, char *commit_GPU, int* bufferBlock_1024, int *bufferBlock_1024_1024, int *threshold, int n)
{
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1024,1,1);
	
	// findMin_kernel
	if ( n>1024*1024 ){
		dimGrid.x = ( n%1024==0 ) ? (n/1024) : (n/1024+1);
		findMin1_kernel<<<dimGrid,dimBlock>>>(costArray_GPU, commit_GPU, bufferBlock_1024_1024, threshold, n);
		n = dimGrid.x;
		dimGrid.x = ( n%1024==0 ) ? (n/1024) : (n/1024+1);		
		findMin2_kernel<<<dimGrid,dimBlock>>>(bufferBlock_1024, bufferBlock_1024_1024, n);
		n = dimGrid.x;
		findMin3_kernel<<<1,dimBlock>>>(bufferBlock_1024, n, threshold);
	}
	else{
		dimGrid.x = ( n%1024==0 ) ? (n/1024) : (n/1024+1);		
		findMin1_kernel<<<dimGrid,dimBlock>>>(costArray_GPU, commit_GPU, bufferBlock_1024, threshold, n);
		n = dimGrid.x;
		findMin3_kernel<<<1,dimBlock>>>(bufferBlock_1024, n, threshold);
	}
	
	//cudaMemcpy( threshold, &minValue, sizeof(int), cudaMemcpyHostToDevice );
	return 1;
}
