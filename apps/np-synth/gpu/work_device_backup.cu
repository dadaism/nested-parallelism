#define N 1

#ifdef ARITH_INTENSE
__device__ inline FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1, v2 = 1;
	v1 = data[idx];
	for (int i=0; i<N; ++i) {
		v1 += v2;
		v2 += v1;
		//v1 *= v2;
		//v2 *= v1;
		v1 += v2;
		v2 += v1;
	}
	value = v2;
	return value;
}

#elif MIX_ARITH_IO
__device__ inline FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1 = 1;
	FLOAT_T v2[N+1] = {1};
	v1 = data[idx];
	for (int i=0; i<N; ++i) {
		v1 += v2[i];
		v2[i] += v1;
		//v1 *= v2[i];
		//v2[i] *= v1;
		v1 += v2[i];
		v2[i] += v1;
		v2[i+1] = v2[i];
	}
	value = v2[N];
	return value;
}

#elif IO_INTENSE
__device__ inline FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	FLOAT_T value = 0.0;
	FLOAT_T v1[N+1];
	FLOAT_T v2[N+1] = {1};
	v1[0] = data[idx];
	for (int i=0; i<N; ++i) {
		v1[i] += v2[i];
		v2[i] += v1[i];
		//v1[i] *= v2[i];
		//v2[i] *= v1[i];
		v1[i] += v2[i];
		v2[i] += v1[i];
		v1[i+1] = v1[i];
		v2[i+1] = v2[i];
	}
	value = v2[N];
	return value;
}
#else	// used for testing
__device__ inline FLOAT_T work(FLOAT_T *data, FLOAT_T *rst, int idx)
{
	return 1.0;
}
#endif
