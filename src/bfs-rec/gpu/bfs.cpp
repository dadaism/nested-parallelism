
#include "bfs.h"
using namespace std;
#include <queue>


void graph_to_dot(graph_t graph, FILE *file){
	if (graph.num_nodes==0){
		printf("empty graph!\n");
		return;
	}
	else{
		fprintf(file, "digraph graph {\n");
		for (unsigned node=0; node<graph.num_nodes; node++){
			fprintf(file, " %llu [shape=circle,label=\"%llu",node,node);
                        fprintf(file,"-%llu\"];\n",graph.levelArray[node]);
		
			for (unsigned edge=graph.vertexArray[node]; edge<graph.vertexArray[node+1];edge++){
				fprintf(file, "%llu -> %llu;\n", node, graph.edgeArray[edge]);
			}
		}
		fprintf(file, "}\n");
			
	}
}

void bfs_rec(graph_t *graph, queue<node_t> *workingSet){
	if (workingSet->empty()) return; 
	node_t node = workingSet->front();
	workingSet->pop();
	unsigned next_level=graph->levelArray_rec[node]+1;
	if (graph->vertexArray[node]!=graph->vertexArray[node+1]){
		for(node_t edge=graph->vertexArray[node]; edge<graph->vertexArray[node+1];edge++){
			node_t neighbor = graph->edgeArray[edge];
			if (graph->levelArray_rec[neighbor]==UNDEFINED || graph->levelArray_rec[neighbor]>next_level){
				graph->levelArray_rec[neighbor]=next_level;
				workingSet->push(neighbor);	
			}
		}
	}
	if (!workingSet->empty()) bfs_rec(graph, workingSet);
}

void bfs_rec(graph_t *graph){
	queue<node_t> workingSet;
	graph->levelArray_rec[graph->source]=0;
	workingSet.push(graph->source);
	bfs_rec(graph, &workingSet);
}

void bfs(graph_t *graph){
	graph->levelArray[graph->source]=0;
	queue<node_t> workingSet;
	workingSet.push(graph->source);
	
	while(!workingSet.empty()){
		node_t node = workingSet.front();
		workingSet.pop();
		unsigned next_level = graph->levelArray[node]+1;
		
		for(node_t edge=graph->vertexArray[node]; edge<graph->vertexArray[node+1];edge++){
			node_t neighbor=graph->edgeArray[edge];
			if (graph->levelArray[neighbor]==UNDEFINED || graph->levelArray[neighbor]>next_level){
				graph->levelArray[neighbor]=next_level;
				workingSet.push(neighbor);
			}
		
		}
	}
}
