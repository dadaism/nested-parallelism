#ifndef __NP_SYNTH_H__
#define __NP_SYNTH_H__

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <vector>
#include <list>

#include "global.h"

typedef float FLOAT_T;

typedef struct conf {
	bool verbose;
	bool debug;
	int solution;
	int device_num;
	char *data_file;
} CONF;

extern CONF config;

extern double init_time;
extern double d_malloc_time;
extern double h2d_memcpy_time;
extern double ker_exe_time;
extern double d2h_memcpy_time;

extern int data_size;
extern int *iter;
extern FLOAT_T *rst;
extern FLOAT_T *data;
extern int VERBOSE;
extern int DEBUG;
extern FILE *fp;

void NP_SYNTH_GPU();

#endif
