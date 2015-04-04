#define N 1
#define FLOAT_T float

__device__ FLOAT_T work_0(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<N; ++i) {
		v1 += v2;
		v2 += v1;
		v1 += v2;
		v2 += v1;
		v1 += v2;
		v2 += v1;
	}
	value = v2;
	return value;
}

__device__ FLOAT_T work_1(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<N; ++i) {
		v1 += v2;
		rst[threadIdx.x] = v1;
		v2 += v1;
		rst[threadIdx.x+1] = v2;
		v1 += v2;
		v2 += v1;
	}
	value = v2;
	return value;
}

__device__ FLOAT_T work_2(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<N; ++i) {
		rst[threadIdx.x] = v1;
		v1 += v2;
		rst[threadIdx.x+1] = v1;
		rst[threadIdx.x+2] = v2;
		v2 += v1;
		rst[threadIdx.x+3] = v2;
	}
	value = v2;
	return value;
}
__device__ FLOAT_T work_3(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	return 1.0;
}
