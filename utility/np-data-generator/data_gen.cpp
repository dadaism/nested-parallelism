#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

typedef struct conf {
    bool verbose;
	int low;
	int high;
    int dist;
	long num;
	float k;
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
    fprintf(stderr,"Usage:  np-data-gen [option]\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "    --help,-h      print this message\n");
    fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
    fprintf(stderr, "\nOther:\n");
    fprintf(stderr, "    --export,-e <data_file>          export data file\n");
    fprintf(stderr, "    --low,-l <number>                specify the minimum value (default: 10)\n");
    fprintf(stderr, "    --high,-m <number>               specify the maximum value (default: 1000)\n");
    fprintf(stderr, "    --num,-n <number>                specify the number of data points (default: 10)\n");
    fprintf(stderr, "    --k,-k <float number>            specify the distribution curve (default: 0)\n");
    fprintf(stderr, "    --dist,-d <number>               specify the distribution (default: 0)\n");
    fprintf(stderr, "                0 - unified distribution\n");
    fprintf(stderr, "                1 - power law distribution\n");
}

void print_conf() { 
    fprintf(stderr, "\nCONFIGURATION:\n");
    if (config.data_file) {
        fprintf(stderr, "- Data file: %s\n", config.data_file);
    }   
    if (config.verbose) fprintf(stderr, "- verbose mode\n");
}

void init_conf() {
	config.num = 10;
    config.low = 10;
    config.high = 1000;
	config.dist = 0;
	config.k = -0.5;
    config.verbose = false;
	config.data_file = NULL;
}

int parse_arguments(int argc, char** argv) {
    int i = 1;
    if ( argc<2 ) { 
        usage();
        return 0;
    }   
    while ( i<argc ) { 
        if ( strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--help")==0 ) { 
            usage();
            return 0;
        }   
        else if ( strcmp(argv[i], "-l")==0 || strcmp(argv[i], "--low")==0 ) { 
            ++i;
            config.low = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-m")==0 || strcmp(argv[i], "--high")==0 ) { 
            ++i;
            config.high = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) { 
            VERBOSE = config.verbose = 1;
        }   
        else if ( strcmp(argv[i], "-d")==0 || strcmp(argv[i], "--dist")==0 ) { 
            ++i;
            config.dist = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-n")==0 || strcmp(argv[i], "--num")==0 ) { 
            ++i;
            config.num = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-k")==0 || strcmp(argv[i], "--k")==0 ) { 
            ++i;
            config.k = atof(argv[i]);
        }   
        else if ( strcmp(argv[i], "-e")==0 || strcmp(argv[i], "--export")==0 ) { 
            ++i;
            if (i==argc) {
                fprintf(stderr, "Data file name missing.\n");
            }   
            config.data_file = argv[i];
        }   
        ++i;
    }   
    
    return 1;
}

inline int unified_dist(int l, int h) {
	srand( gettime_usec() );
	return rand() % (h-l) + l;
}

inline int power_law(int l, int h, float k) {
	srand( gettime_usec() );
	double value = 	( pow(h, k+1) - pow(l, k+1) ) * ( (double)rand() / RAND_MAX) + pow(l, k+1);
	return pow(value, 1.0/(k+1));
}

void gen_data() {
	int y = 0;
 	fprintf(fp, "%ld\n", config.num);
	for (long i=0; i<config.num; ++i) {
		switch ( config.dist ) {
			case 0: y = unified_dist( config.low, config.high ); 
				break;
			case 1: y = power_law( config.low, config.high, -config.k ) ;
				break;
			default:
				break;
		}
		fprintf(fp, "%d ", y);
	}
	fprintf(fp, "\n");
}

int main(int argc, char *argv[])
{
	init_conf();
    if ( !parse_arguments(argc, argv) ) return 0;
    
	if (VERBOSE)
    	print_conf();
    
    if ( config.data_file!=NULL ) {
        fp = fopen(config.data_file, "w");
        if ( fp==NULL ) { 
            fprintf(stderr, "Error: NULL file pointer.\n");
            return 1;
        }   
    }   
    else
		fp = stdout;

    double start_time, end_time;
    
    start_time = gettime();
    
    gen_data();
    
    end_time = gettime();
    if (VERBOSE)
        fprintf(stderr, "Import graph:\t\t%lf\n",end_time-start_time);

    return 0;
}
