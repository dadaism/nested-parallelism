#ifndef __ANALYZER_H__
#define __ANALYZER_H__

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include <vector>
#include <list>

#include "global.h"

struct _INFO_ {
	int noNodeTotal;
	int noEdgeTotal;
	long totalDegree;
	float avgDegree;
	int minDegree;
	int maxDegree;
	long *distDegree;
};


typedef struct conf {
    bool verbose;
    bool dist;
    int data_set_format;
    int solution;
    char *graph_file;
} CONF;


#endif
