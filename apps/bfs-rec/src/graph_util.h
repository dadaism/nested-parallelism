#ifndef __GRAPH_UTIL_H__
#define __GRAPH_UTIL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <vector>
#include <list>
#include <queue>
#include "bfs.h"

#define BUFF_SIZE 1024000

#define EXIT(msg) \
		fprintf(stderr, "info: %s:%d: ", __FILE__, __LINE__); \
		fprintf(stderr, "%s", msg);	\
		exit(0);

using namespace std;

//graph related variables
list<node_t> *adjacencyNodeList;
list<weight_t> *adjacencyWeightList;
node_t noNodeTotal = 0;
node_t noEdgeTotal = 0;

//buffer for stdin
char buff[BUFF_SIZE] = {};

void delete_graph(graph_t *graph){
	delete [] graph->vertexArray;
	delete [] graph->edgeArray;
	delete [] graph->weightArray;
	delete [] graph->levelArray;
	delete [] graph->levelArray_rec;
	delete [] graph->levelArray_gpu;
	delete [] graph->levelArray_gpu_np;
	delete [] graph->levelArray_gpu_np_hier;
}

int convertCSR(graph_t *graph);

int readInputDIMACS9(graph_t *graph)
{
	char *p;
	node_t tailNode, headNode;
	weight_t edgeWeight;
	graph->source=0;
	while( fgets(buff, BUFF_SIZE, stdin) != NULL ){
		if ( buff[0] == 'c' )   continue;
		if ( buff[0] == 'p' ){
			p = strtok(buff+4, " ");
			noNodeTotal = (node_t)atoi(p);
			p = strtok( NULL, " ");
			noEdgeTotal = (node_t)atoi(p);
			//printf(" Number of node: %d\n Number of edge: %d\n", noNodeTotal, noEdgeTotal);
			adjacencyNodeList = new list<node_t> [noNodeTotal];
			adjacencyWeightList = new list<weight_t> [noNodeTotal];
		}
		if ( buff[0] == 's' ){
			p = strtok(buff+1, " ");
			graph->source = (node_t) (atoi(p) - 1);
		}
		if ( buff[0] == 'a' ){
			/* get tail vertex no */
			p = strtok(buff+1, " ");
			tailNode = (node_t) atoi(p) - 1;
			/* get head vertex no */
			p = strtok(NULL, " ");
			headNode = (node_t) atoi(p) - 1;
			/* get edge value no */
			p = strtok(NULL, " ");
			edgeWeight = (weight_t)atoi(p);
			//printf("%d %d %d\n", tailNode, headNode,edgeWeight);
			adjacencyNodeList[tailNode].push_back(headNode);
			adjacencyWeightList[tailNode].push_back(edgeWeight);
		}
	}
	convertCSR(graph);
	return 0;
}

int readInputDIMACS10(graph_t *graph)
{
	char *p;
	node_t tailNode, headNode;
	weight_t edgeWeight;
	graph->source=0;
	while( fgets(buff, BUFF_SIZE, stdin) != NULL ){ // read comments
		if ( buff[0] == '%' )   continue;
		else	break;
	}
	//printf("%d\n", strlen(buff));
	p = strtok(buff," ");
	noNodeTotal = (node_t)atoi(p);
	p = strtok(NULL, " ");
	noEdgeTotal = (node_t)atoi(p);
	noEdgeTotal = 2 * noEdgeTotal;
	//printf(" Number of node: %d\n Number of edge: %d\n", noNodeTotal, noEdgeTotal);

	adjacencyNodeList = new list<node_t> [noNodeTotal];
	adjacencyWeightList = new list<weight_t> [noNodeTotal];
	srand( 7 );
	for (node_t i=0; i<noNodeTotal; ++i){
		tailNode = i;
		if ( fgets(buff, BUFF_SIZE, stdin)==NULL ) {
			EXIT("Parsing adjacencylist fails!");
		}
		if ( strlen(buff) == BUFF_SIZE-1 )
			printf("line: %d\n", i);
		assert ( strlen(buff) != BUFF_SIZE-1 );
		p = strtok(buff, " ");
		while( p!=NULL && (*p)!='\n' ){
			//degree++;
			headNode = (node_t)atoi(p) - 1;
			adjacencyNodeList[tailNode].push_back(headNode);
			weight_t weight = rand() % 1000 + 1;
			adjacencyWeightList[tailNode].push_back(weight);
			p = strtok(NULL, " ");
		}
	}
	convertCSR(graph);
	return 0;
}

int readInputSLNDC(graph_t *graph)
{
	/* input raw data */
	char *p;
	node_t tailNode, headNode;
	weight_t edgeWeight = 1;
	node_t maxNodeNo = 0;
	list<node_t> *tempNodeList;
	graph->source=0;
	node_t *mapTable;
	if ( fgets(buff, 256, stdin)==NULL ) {
		EXIT("");
	}
	if ( fgets(buff, 256, stdin)==NULL ) {
		EXIT("");
	}
	/* parse node number and edge number */
	if ( fgets(buff, 256, stdin)==NULL ) {
		EXIT("Parsing total node number fails!");
	}
	p = strtok(buff+9, " ");
	noNodeTotal = (node_t) atoi(p);
	p = strtok( NULL, " ");		p = strtok( NULL, " ");
	noEdgeTotal = (node_t) atoi(p);
	//printf(" Number of node: %d\n Number of edge: %d\n", noNodeTotal, noEdgeTotal);
	if ( fgets(buff, 256, stdin)==NULL ) {
		EXIT("");
	}

	tempNodeList = new list<node_t> [noNodeTotal*2];
	mapTable = new node_t [noNodeTotal*2] ();
	noEdgeTotal = 0;
	while( scanf("%ld%ld", &tailNode, &headNode)!= EOF ){
		//printf("%d %d %d\n", tailNode, headNode,edgeWeight);
		//printf("%d %d\n", tailNode, headNode);
		if ( tailNode>=noNodeTotal*2 || headNode>=noNodeTotal*2 ) {
			printf("Tail node: %d\nHead node: %d\n", tailNode, headNode);
			exit(0);
		}
		mapTable[tailNode] = mapTable[headNode] = 1;
		if ( tailNode>maxNodeNo )
			maxNodeNo = tailNode;
		if ( headNode>maxNodeNo )
			maxNodeNo = headNode;
		tempNodeList[tailNode].push_back(headNode);
		noEdgeTotal++;
	}

	/* build mapping table */
	noNodeTotal = 0;
	for (node_t i=0; i<=maxNodeNo; ++i){
		if ( mapTable[i]==1 ){
			mapTable[i] = noNodeTotal;
			noNodeTotal++;
		}
		else
			mapTable[i] = -1;
	}

	/* eliminate discrete node */
	adjacencyNodeList = new list<node_t> [noNodeTotal];
	adjacencyWeightList = new list<weight_t> [noNodeTotal];
	printf("Node number is %d\n", noNodeTotal);
	noNodeTotal = 0;
	srand( 7 );
	for (node_t i=0; i<=maxNodeNo; ++i){
		if ( mapTable[i] == -1 )	// no mapping
			continue;
		else{						// convert to contiguous adjacencylist
			tailNode = mapTable[i];
			while ( !tempNodeList[i].empty() ){
				int tmpHeadNode = tempNodeList[i].front();
				tempNodeList[i].pop_front();
				headNode = mapTable[tmpHeadNode];
				if (headNode==-1)
					printf("Error for mapTable, %d\n", tmpHeadNode);
				adjacencyNodeList[tailNode].push_back(headNode);
				weight_t weight = rand() % 1000 + 1;
				adjacencyWeightList[tailNode].push_back(weight);
			}
			noNodeTotal++;
			//printf("Size of node %d is %d\n", tailNode, adjacencyNodeList[tailNode].size() );
		}
	}
	delete [] tempNodeList;
	delete [] mapTable;
	convertCSR(graph);
	return 0;
}

int convertCSR(graph_t *graph)
{
	node_t startingPos, noEdgePerNode;
	printf("Edge no: %d\n", noEdgeTotal);
	graph->num_nodes = noNodeTotal;
	graph->num_edges = noEdgeTotal;
	graph->vertexArray = new node_t [noNodeTotal+1] ();
	graph->edgeArray = new node_t [noEdgeTotal] ();
	graph->weightArray = new weight_t [noEdgeTotal] ();

	graph->levelArray = new unsigned [noNodeTotal]();
	graph->levelArray_rec = new unsigned [noNodeTotal]();
	graph->levelArray_gpu = new unsigned [noNodeTotal]();
	graph->levelArray_gpu_np = new unsigned [noNodeTotal]();
	graph->levelArray_gpu_np_hier = new unsigned [noNodeTotal]();

	startingPos = 0;
	noEdgePerNode = 0;
	for (node_t i=0; i<noNodeTotal; ++i ){
		startingPos = startingPos + noEdgePerNode;
		noEdgePerNode = 0;
		graph->vertexArray[i] = startingPos;
		//printf("Node %d is connected to :", i+1);
		while ( adjacencyNodeList[i].empty()!=true && adjacencyWeightList[i].empty()!=true ){
			graph->edgeArray[ startingPos + noEdgePerNode ] = adjacencyNodeList[i].back();
			graph->weightArray[ startingPos + noEdgePerNode ] =  adjacencyWeightList[i].back();
			adjacencyNodeList[i].pop_back();
			adjacencyWeightList[i].pop_back();
			//printf("%d(%d)\t",graph->edgeArray[startingPos+noEdgePerNode]+1, graph->weightArray[startingPos+noEdgePerNode]);
			noEdgePerNode++;	
		}
		graph->levelArray[i]=UNDEFINED;
		graph->levelArray_rec[i]=UNDEFINED;
		graph->levelArray_gpu[i]=UNDEFINED;
		graph->levelArray_gpu_np[i]=UNDEFINED;
		graph->levelArray_gpu_np_hier[i]=UNDEFINED;
		//printf("\n");
	}
	graph->vertexArray[noNodeTotal] = noEdgeTotal;
	graph->levelArray[graph->source]=0;
	graph->levelArray_rec[graph->source]=0;
	graph->levelArray_gpu[graph->source]=0;
	graph->levelArray_gpu_np[graph->source]=0;
	graph->levelArray_gpu_np_hier[graph->source]=0;
	delete [] adjacencyNodeList;
	delete [] adjacencyWeightList;
	printf("source=%ld\n", graph->source);
	return 0;

}

#endif
