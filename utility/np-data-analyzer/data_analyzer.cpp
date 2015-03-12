#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#define MAX_ITER_SIZE 999999

typedef struct conf {
    bool verbose;
	int low;
	int high;
    int dist;
	long num;
	long *hist;
    char *data_file;
} CONF;

CONF config;
bool VERBOSE = false;
FILE *fp = NULL;

double gettime() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int gettime_usec() {
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec*1e+6+t.tv_usec;
}

void usage() {
    fprintf(stderr,"\n");
    fprintf(stderr,"Usage:  np-data-analyzer [option]\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "    --help,-h      print this message\n");
    fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
    fprintf(stderr, "\nOther:\n");
    fprintf(stderr, "    --import,-i <data_file>       export data file\n");
}

void print_conf() { 
    fprintf(stderr, "\nCONFIGURATION:\n");
    if (config.data_file) {
        fprintf(stderr, "- Data file: %s\n", config.data_file);
    }   
    if (config.verbose) fprintf(stderr, "- verbose mode\n");
}

void init_conf() {
	config.num = 0;
    config.low = 0;
    config.high = 0;
	config.hist = (long *)malloc(MAX_ITER_SIZE*sizeof(long));
	for (int i=0; i<MAX_ITER_SIZE; ++i)
		config.hist[i] = 0;
    config.verbose = false;
	config.data_file = NULL;
}

int parse_arguments(int argc, char** argv) {
    int i = 1;
    if ( argc<3 ) { 
        usage();
        return 0;
    }
    while ( i<argc ) { 
        if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) { 
            usage();
            return 0;
        }   
        else if ( strcmp(argv[i], "-i")==0 || strcmp(argv[i], "--import")==0 ) { 
            ++i;
            if (i==argc) {
                fprintf(stderr, "Data file name missing.\n");
            }   
            config.data_file = argv[i];
        }
		else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) {
 			VERBOSE = config.verbose = 1;
 		}
        ++i;
    }   
    return 1;
}

void analyze_data() {
	int iter_size = 0; 
	fscanf(fp, "%d\n", &config.num);
 	
    config.low = 0;
    config.high = 0;
	
	long long total = 0;
	fprintf(stdout, "Total nodes: %d\n", config.num);	
	for (long i=0; i<config.num; ++i) {
		fscanf(fp, "%d", &iter_size);
		if ( config.low>iter_size )	config.low = iter_size;
		if ( config.high<iter_size )	config.high = iter_size;
		config.hist[iter_size] += 1;
		total += iter_size;
	}
	fprintf(stdout, "Total: %ld\n", total);	
	fprintf(stdout, "Max: %d\n", config.high);	
	fprintf(stdout, "Min: %d\n", config.low);	
	fprintf(stdout, "Avg: %.2f\n", (float)total/config.num);	
	if (VERBOSE)
		for (long i=config.low; i<=config.high; ++i) {
			fprintf(stdout, "%d %d\n", i, config.hist[i]);
		}
}

int main(int argc, char *argv[])
{
	init_conf();
    if ( !parse_arguments(argc, argv) ) return 0;
    
	if (VERBOSE)
    	print_conf();
    
    if ( config.data_file!=NULL ) {
        fp = fopen(config.data_file, "r");
        if ( fp==NULL ) { 
            fprintf(stderr, "Error: NULL file pointer.\n");
            return 1;
        }   
    }   
    else
		fp = stdin;

    double start_time, end_time;
   
    start_time = gettime();
    
	analyze_data();

    end_time = gettime();
    if (VERBOSE)
        fprintf(stderr, "Analyze graph:\t\t%lf\n",end_time-start_time);

    return 0;
}
