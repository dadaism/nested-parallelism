#define M 50

#ifdef ARITH_INTENSE
__device__ FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<M; ++i) {
		v1 += v2;
		v2 += v1;
	}
	value = v2;
	return value;
}

#elif MIX_ARITH_IO
/*__device__ FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<M; ++i) {
		v1 += v2;
		rst[threadIdx.x+i] = v1;
		v2 += v1;
	}
	value = v2;
	return value;
}*/
__device__ FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<threadIdx.x; ++i) {
		v1 += v2;
		rst[threadIdx.x+i] = v1;
		v2 += v1;
	}
	value = v2;
	return value;
}


#elif IO_INTENSE
__device__ FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<M; ++i) {
		rst[threadIdx.x+i] = v1;
		v1 += v2;
		rst[threadIdx.x+i+1] = v1;
		rst[threadIdx.x+i+2] = v2;
		v2 += v1;
		rst[threadIdx.x+i+3] = v2;
	}
	value = v2;
	return value;
}

#else
__device__ FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	return 1.0;
}
#endif
