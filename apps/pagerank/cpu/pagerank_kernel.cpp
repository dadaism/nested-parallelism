#include <stdio.h>
#include "pagerank.h"

#define DAMPING_FACTOR 0.5
#define EPSILON 1.0e-13

//char *update = new char [noNodeTotal] ();
//int *queue = new int [queue_max_length];
//unsigned int queue_max_length = QMAXLENGTH;
//unsigned int queue_length = 0;
unsigned int nonstop = 0;

// generate queue for dangling nodes
int noDanglingNode = 0;
int *danglingVertexArray;
FLOAT_T *danglingRankArray;
 
FLOAT_T dangling_rank = 0.0;
FLOAT_T old_dangling_rank = 0.0;
FLOAT_T damping = DAMPING_FACTOR;
FLOAT_T rank_random_walk;
FLOAT_T rank_dangling_node;
FLOAT_T delta_rank_dangling_node;

FLOAT_T *newRankArray;

double start_time, end_time;
double dangling_start;

void prepare_dangling_node()
{
	noDanglingNode = 0;
	// generate queue for dangling nodes
	for (int i=0; i<noNodeTotal; ++i) {
		int start = graph.vertexArray[i];
		int end = graph.vertexArray[i+1];
		if ( start==end )
			noDanglingNode++;
    }   
    if ( noDanglingNode!=0 ) {
        danglingVertexArray = new int [noDanglingNode] (); 
        danglingRankArray = new FLOAT_T [noDanglingNode] ();
        int index = 0;
        for (int i=0; i<noNodeTotal; ++i) {
            int start = graph.vertexArray[ i ];
            int end = graph.vertexArray[ i+1 ];
            if ( start==end )
                danglingVertexArray[index++] = i;
		}   
	}

	rank_random_walk = (1-damping)*TOTAL_RANK/noNodeTotal;
}


double cal_danglingRank(int *danglingVertexArray, int noDanglingNode)
{
    double danglingRank = 0.0; 
    for (int i=0; i<noDanglingNode; ++i) {
        int id = danglingVertexArray[i];
        danglingRank += graph.rankArray[ id ];
    }    
    return danglingRank;
}

void clean_cpu()
{
	if ( noDanglingNode!=0 )
		delete [] danglingVertexArray;
	delete [] newRankArray;
}

void pg_pull_cpu()
{
	bool stop = false;
	int iteration = 0;
	FLOAT_T damping = DAMPING_FACTOR;
	FLOAT_T danglingRank = 0.0;
	FLOAT_T *tmp;
	FLOAT_T totalRank = 0;
	FLOAT_T epsilon = EPSILON;
	
	newRankArray = new FLOAT_T [noNodeTotal] ();
	// Set omp enviroment
	omp_set_dynamic(0);
	omp_set_num_threads(config.thread_num);
	// generate queue for dangling nodes

	while ( !stop ) {
	//while ( !stop ) {
		iteration++;
		stop = true;
		memset(newRankArray, 0, sizeof(FLOAT_T)*noNodeTotal);
		// Process dangling node
		danglingRank = cal_danglingRank(danglingVertexArray, noDanglingNode);
		
		#pragma omp parallel for
		for ( int curr=0; curr<noNodeTotal; ++curr ) {
			// Pull from neighbours
			int start = graph.childVertexArray[ curr ];
			int end = graph.childVertexArray[ curr+1 ];
			//printf("Start: %d\tEnd: %d\n", start, end);
			for (int i=start; i<end; ++i){
				int parent = graph.rEdgeArray[i];
				//printf("Parent node: %d\n", parent);
				double rank = graph.rankArray[ parent ] / graph.outdegreeArray[ parent ];
        		newRankArray[curr] = newRankArray[curr] + damping * rank;
				//printf("node %d pull from node %d, rank = %f\n", curr, parent, rank);
			}
			newRankArray[ curr ] += (1-damping) / noNodeTotal; 
			newRankArray[ curr ] += damping * danglingRank / noNodeTotal; 
		}
		tmp = graph.rankArray;
		graph.rankArray = newRankArray;
		newRankArray = tmp;
		//queueSize = 0;
		#pragma omp parallel for
		for ( int i=0; i<noNodeTotal; ++i) {
			double ep = fabs( graph.rankArray[i] - newRankArray[i] );
			if ( ep>EPSILON) {
				//queueSize++;
				stop = false;
			}
		}
	}// end of while( iteration )
	printf("PageRank ends in %d iterations\n", iteration);


}

void PAGERANK_CPU()
{
	prepare_dangling_node();

	start_time = gettime();
	switch (config.solution) {
		case 0:  pg_pull_cpu();	// 
			break;
		default:
			break;
	}
    
	end_time = gettime();
	ker_exe_time += end_time - start_time;
	
	clean_cpu();
}

