#ifndef __BFS_REC_H__
#define __BFS_REC_H__

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <vector>
#include <list>

#include "global.h"

typedef struct conf {
	bool verbose;
	bool debug;
	int data_set_format;
	int solution;
	int device_num;
	char *graph_file;
} CONF;

extern CONF config;

void BFS_REC_GPU();

#endif
