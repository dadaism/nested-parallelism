#include "pagerank.h"

inline int atomicInc(int *ptr, int val) {
	return __sync_fetch_and_add(ptr, val);
}

int *generate_danglingVertexArray( int *noDanglingNode )
{   int *danglingVertexArray;
    for (int i=0; i<noNodeTotal; ++i) {
        if ( graph.outdegreeArray[i]==0 ) 
            (*noDanglingNode)++;
    }    
    if ( noDanglingNode!=0 ) {
        danglingVertexArray = new int [(*noDanglingNode)] ();
        int index = 0; 
        for (int i=0; i<noNodeTotal; ++i) {
            if ( graph.outdegreeArray[i]==0 ) 
                danglingVertexArray[index++] = i; 
        }    
        printf("Dangling vertex in total: %d\n", index);
    }    
    return danglingVertexArray;
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


int PageRank_OMP_PULL()
{
	bool stop = false;
	int iteration = 0;
	int queueSize = noNodeTotal;
	double damping = DAMPING_RATIO;
	double danglingRank = 0.0;
	double *tmp;
	double *newRankArray = new double [noNodeTotal] ();
	double totalRank = 0;
	double epsilon = 1;
	
	// Set omp enviroment
	omp_set_num_threads(config.thread_num);
	// generate queue for dangling nodes
	int noDanglingNode = 0;
	int *danglingVertexArray;
	danglingVertexArray = generate_danglingVertexArray( &noDanglingNode );

	//while ( epsilon > 1.0e-10 ) {
	//while ( queueSize>0 ) {
	//while ( iteration<1 ) {
	while ( !stop ) {
		iteration++;
		stop = true;
		epsilon = 0;
		memset(newRankArray, 0, sizeof(double)*noNodeTotal);
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
	if ( noDanglingNode!=0 )
		delete [] danglingVertexArray;
	delete [] newRankArray;

	return 0;
}

