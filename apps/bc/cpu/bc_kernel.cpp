#include "bc.h"

void forward_bfs_init(int source) 
{
	cont = true;
	memset(p, 0, sizeof(char)*noNodeTotal*noNodeTotal);
	
	#pragma omp parallel for
	for (int i=0; i<noNodeTotal; ++i) {
		graph.frontier[i] = 0;
		graph.levelArray[i] = MAX_LEVEL;
		sigma[i] = 0;
	}
	graph.frontier[source] = 1;
	graph.levelArray[source] = 0;
	sigma[source] = 1;
}


void bitmap_workset()
{
	#pragma omp parallel for
	for (int curr=0; curr<noNodeTotal; ++curr) {
		graph.frontier[ curr ] = graph.update[ curr ];
		graph.update[ curr ] = 0;
	}
}

void forward_bfs_bitmap(int dist) {
	#pragma omp parallel for
	for ( int curr=0; curr<noNodeTotal; ++curr) {
		if ( graph.frontier[curr] == 0 )
			continue;
		int start = graph.vertexArray[ curr ];
		int end = graph.vertexArray[ curr+1 ];
		for (int i=start; i<end; ++i) {
			int nid = graph.edgeArray[ i ];
			if ( graph.levelArray[ nid ] == MAX_LEVEL ) {
				graph.levelArray[ nid ] = dist + 1;
				graph.update[ nid ] = 1;
				cont = true;
			}
			if ( graph.levelArray[ nid ] == dist+1 ) {
				p[curr*noNodeTotal+nid] = 1;	// p[curr][nid] = 1;
				atomicInc( &sigma[nid], sigma[curr] );
			}
		}
	}
}

void backward_propagation(int dist) 
{
	#pragma omp parallel for
	for ( int curr=0; curr<noNodeTotal; ++curr) {
		int start = graph.vertexArray[ curr ];
		int end = graph.vertexArray[ curr+1 ];
		for (int i=start; i<end; ++i) {
			int nid = graph.edgeArray[ i ];
			if ( graph.levelArray[nid]==dist-1 && p[curr*noNodeTotal+nid]==1 ) {
				delta[curr] = delta[curr] + (double)sigma[curr]/sigma[nid]*(1+delta[nid]);
			}
		}
	}
}

void old_backward_propagation(int dist) 
{
	/* Initialization */
	#pragma omp parallel for
	for ( int curr=0; curr<noNodeTotal; ++curr) {
		if ( graph.levelArray[curr]==dist-1 ) {
			for (int nid=0; nid<noNodeTotal; ++nid) {
				if ( p[nid*noNodeTotal+curr]==1 ) {	// p[nid][curr]
					delta[nid] = delta[nid] + (double)sigma[nid]/sigma[curr]*(1+delta[curr]);
					//delta[nid] = delta[nid] + delta[curr];
				}

			}
		}
	}
}

void backward_sum(int source, int dist) 
{
	#pragma omp parallel
	{
		#pragma omp for nowait
		for ( int i=0; i<source; ++i) {
			if ( graph.levelArray[i]==dist-1)
				bc[i] = bc[i] + delta[i];	
				//bc[i] = bc[i] + delta[i]*sigma[i]-1;	
		}
		
		#pragma omp for nowait
		for (int i=source+1; i<noNodeTotal; ++i ) {
			if ( graph.levelArray[i]==dist-1 )
				bc[i] = bc[i] + delta[i];
				//bc[i] = bc[i] + delta[i]*sigma[i]-1;
		}
	}
}

void backward_init()
{
	memset(	delta, 0, sizeof(float)*noNodeTotal );
}
