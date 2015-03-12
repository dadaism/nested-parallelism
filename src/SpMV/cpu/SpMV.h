#ifndef __SPMV_H__
#define __SPMV_H__

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <vector>
#include <list>

#include "global.h"

typedef double FLOAT_T;

typedef struct conf {
	bool verbose;
	bool debug;
	int data_set_format;
	int solution;
	int thread_num;
	char *graph_file;
} CONF;

extern CONF config;

extern FLOAT_T*data;
extern FLOAT_T *x;
extern FLOAT_T *y;

extern double start_time;
extern double end_time;
extern double init_time;
extern double d_malloc_time;
extern double h2d_memcpy_time;
extern double ker_exe_time;
extern double d2h_memcpy_time;

void SpMV_CPU();

#endif
