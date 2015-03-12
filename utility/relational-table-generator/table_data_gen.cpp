#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#define KEY_RANGE 6666

typedef struct conf {
    bool verbose;
    int dist;
	float k;
    char *data_file;
} CONF;

typedef struct tb {
	int size;
	short *key;
	short *value;
} TABLE;

CONF config;
bool VERBOSE = false;
FILE *fp = NULL;

TABLE table1;
TABLE table2;

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
    fprintf(stderr,"Usage:  table-data-gen [option]\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "    --help,-h      print this message\n");
    fprintf(stderr, "    --verbose,-v   basic verbosity level\n");
    fprintf(stderr, "\nOther:\n");
    fprintf(stderr, "    --export,-e <data file>          export data file\n");
    fprintf(stderr, "    --size1 <number>                 specify the size of 1st table\n");
    fprintf(stderr, "    --size2 <number>                 specify the size of 2nd table\n");
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
	config.dist = 0;
	config.k = -0.5;
    config.verbose = false;
	config.data_file = NULL;

	table1.size = 10;
	table2.size = 10;
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
        else if ( strcmp(argv[i], "--size1")==0 ) { 
            ++i;
            table1.size = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "--size2")==0 ) { 
            ++i;
            table2.size = atoi(argv[i]);
        }   
        else if ( strcmp(argv[i], "-v")==0 || strcmp(argv[i], "--verbose")==0 ) { 
            VERBOSE = config.verbose = 1;
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

void gen_table_data() {
	short start = 1;
	short curr = 1;
	short end = 1;
	table1.key = new short [table1.size];
	table1.value = new short [table1.size];
	table2.key = new short [table2.size];
	table2.value = new short [table2.size];
	// generate relational table 1 (ordered key)
	for (int i=0; i<table1.size; ++i) {
		int increment = rand()%3 + 1;
		curr = curr + increment;
		table1.key[i] = curr;
		table1.value[i] = rand() % KEY_RANGE + 1;
	}
	end = curr + 1;

	// generate relational table 2 
	for (int i=0; i<table2.size; ++i) {
		table2.key[i] = unified_dist(start, end); 	// can switch to pow-law distribution
		table2.value[i] = rand() % KEY_RANGE + 1;
	}

}

void dump_data() {

	// dump relational table 1 (ordered key)
	fprintf(fp, "%d\n", table1.size);
	for (int i=0; i<table1.size; ++i) {
		fprintf(fp, "%d %d\n", table1.key[i], table1.value[i]);
	}

	// dump relational table 2 
	fprintf(fp, "%d\n", table2.size);
	for (int i=0; i<table2.size; ++i) {
		fprintf(fp, "%d %d\n", table2.key[i], table2.value[i]);
	}


}

void destroy_table_data() {
	delete [] table1.key;
	delete [] table1.value;
	delete [] table2.key;
	delete [] table2.value;
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
    
    gen_table_data();
 
	dump_data();   
    end_time = gettime();
    if (VERBOSE)
        fprintf(stderr, "Import graph:\t\t%lf\n",end_time-start_time);

    return 0;
}
