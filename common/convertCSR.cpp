#include "convertCSR.h"

int convertCSR()
{
	int startingPos, noEdgePerNode;
	graph.vertexArray = new int [noNodeTotal+1] ();
	graph.costArray = new int [noNodeTotal] ();
	graph.edgeArray = new int [noEdgeTotal] ();
	graph.weightArray = new int [noEdgeTotal] ();
	graph.frontier = new char [noNodeTotal] ();
	//graph.visited = new char [noNodeTotal] ();
	
	startingPos = 0;
	noEdgePerNode = 0;
	for (int i=0; i<noNodeTotal; ++i ){
		startingPos = startingPos + noEdgePerNode;
		noEdgePerNode = 0;
		graph.vertexArray[i] = startingPos;
		//printf("Node %d is connected to :", i+1);
		while ( adjacencyNodeList[i].empty()!=true && adjacencyWeightList[i].empty()!=true ){
			graph.edgeArray[ startingPos + noEdgePerNode ] = adjacencyNodeList[i].back();
			graph.weightArray[ startingPos + noEdgePerNode ] =  adjacencyWeightList[i].back();
			adjacencyNodeList[i].pop_back();
			adjacencyWeightList[i].pop_back();
			//printf("%d(%d)\t",graph.edgeArray[startingPos+noEdgePerNode]+1, graph.weightArray[startingPos+noEdgePerNode]);
			noEdgePerNode++;	
		}
		//printf("\n");
	}
	graph.vertexArray[noNodeTotal] = noEdgeTotal;
	delete [] adjacencyNodeList;
	delete [] adjacencyWeightList;
	return 0;

}
