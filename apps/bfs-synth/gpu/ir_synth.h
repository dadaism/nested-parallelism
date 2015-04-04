#ifndef __IR_SYNTH_H__
#define __IR_SYNTH_H__

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

void IR_SYNTH_GPU();

#endif
