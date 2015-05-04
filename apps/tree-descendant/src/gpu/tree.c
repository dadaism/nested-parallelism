#include "tree.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

void gen_regular_tree(tree_t *tree, unsigned num_levels, unsigned outdegree){
	tree->num_levels = num_levels;
	tree->num_nodes = 0;
	for (unsigned level=0; level<tree->num_levels;level++){
		tree->num_nodes += pow(outdegree,level);
	}
	printf("gen_regular_tree:: num_levels=%u, outdegree=%u, number of nodes=%llu.\n", num_levels, outdegree, tree->num_nodes);
	if (tree->num_nodes==0) return;
	tree->vertexArray = new node_t[tree->num_nodes+1];
	tree->parentArray = new node_t [tree->num_nodes];
	tree->levelArray = new unsigned[tree->num_nodes];
	tree->descendantArray = new node_t [tree->num_nodes];
	tree->descendantArray_rec = new node_t [tree->num_nodes];
	tree->descendantArray_gpu = new node_t [tree->num_nodes];
	tree->descendantArray_gpu_np = new node_t [tree->num_nodes];
	tree->descendantArray_gpu_np_hier = new node_t [tree->num_nodes];
	tree->edgeArray = new node_t [tree->num_nodes];
	
	node_t last_created=0;
	node_t edge_pointer=0;
	tree->parentArray[0]=(node_t)-1;
	tree->levelArray[0]=0;
	for (node_t node=0; node < tree->num_nodes; node+=1){
		tree->vertexArray[node]=edge_pointer;
		tree->descendantArray[node]=1;
		tree->descendantArray_rec[node]=1;
		tree->descendantArray_gpu[node]=1;
		tree->descendantArray_gpu_np[node]=1;
		tree->descendantArray_gpu_np_hier[node]=1;
		//if node is not a leaf, create the children
		if (tree->levelArray[node] < (num_levels-1)){
			for (int i=0;i<outdegree;i++){
				tree->edgeArray[edge_pointer++]=++last_created;
				tree->parentArray[last_created]=node;	
				tree->levelArray[last_created]=tree->levelArray[node]+1;
			}
		}
	}
	tree->vertexArray[tree->num_nodes]=edge_pointer;
} 

inline unsigned randint(unsigned lo, unsigned hi) { return lo + ((unsigned)rand() % (hi + 1 - lo)); } //random

void gen_random_tree(tree_t *tree, unsigned height_min, unsigned height_max, unsigned outdegree_min, unsigned outdegree_max, unsigned possibility){

	srand(0);

	unsigned max_nodes = 280000000; //maximum size of array that can be put on a K20 GPU	
//	for (unsigned level=0; level<height_max; level++){
//		max_nodes += pow(outdegree_max,level) ;
//	}
//	printf("max nodes = %llu \n",max_nodes);

	unsigned *child_nodes = new unsigned[max_nodes]; //nodes[i] stores the number of children of node i
       
	if (child_nodes==NULL) {printf("Memory allcoate failed"); exit(0);}
	
	printf("child nodes array allocate sucessful.\n");
	
	unsigned children = randint(outdegree_min, outdegree_max);
	node_t nodes_at_level = (node_t) children;
	node_t nodes_at_next_level;
	node_t total_nodes = 1;
	child_nodes[0]=children;

	for (unsigned level = 1; level < height_max; level++){
		nodes_at_next_level=0;
		int must_have_child=randint(0,nodes_at_level-1);
		for (node_t n=0; n<nodes_at_level;n++){
		     if(level < height_min){
			children = (level==height_max-1) ? 0 : randint(outdegree_min, outdegree_max);
			child_nodes[total_nodes++]=children;

			if (total_nodes==max_nodes) {printf("Memory allcoate failed at level: %d, node : %d",level, n); exit(0);}


			nodes_at_next_level+=(node_t)children;
		     } else if(n==must_have_child){
			children = (level==height_max-1) ? 0 : randint(outdegree_min, outdegree_max);
			child_nodes[total_nodes++]=children;

			if (total_nodes==max_nodes) {printf("Memory allcoate failed at level: %d, node : %d",level, n); exit(0);}



			nodes_at_next_level+=(node_t)children;
		     } else {
			unsigned if_has_child_node=1;
			unsigned iters=0;
			while( iters++<possibility)
				if_has_child_node*=randint(0,1);
			if (if_has_child_node) {
				children = (level==height_max-1) ? 0 : randint(outdegree_min, outdegree_max); 
				child_nodes[total_nodes++]=children;

				if (total_nodes==max_nodes) {printf("Memory allcoate failed at level: %d, node : %d",level, n); exit(0);}


				nodes_at_next_level+=(node_t)children;
			}else{
				child_nodes[total_nodes++]=0;

				if (total_nodes==max_nodes) {printf("Memory allcoate failed at level: %d, node : %d",level, n); exit(0);}


			}

		     }		     
		}
		nodes_at_level = nodes_at_next_level;
	}

	printf("Total nodes: %u\n",total_nodes);

	tree->num_levels = height_max;
	tree->num_nodes = total_nodes;
	
	tree->vertexArray = new node_t[tree->num_nodes+1];
	tree->parentArray = new node_t [tree->num_nodes];
	tree->levelArray = new unsigned[tree->num_nodes];
	tree->descendantArray = new node_t [tree->num_nodes];
	tree->descendantArray_rec = new node_t [tree->num_nodes];
	tree->descendantArray_gpu = new node_t [tree->num_nodes];
	tree->descendantArray_gpu_np = new node_t [tree->num_nodes];
	tree->descendantArray_gpu_np_hier = new node_t [tree->num_nodes];
	tree->edgeArray = new node_t [tree->num_nodes];
	

	node_t last_created=0;
	node_t edge_pointer=0;
	tree->parentArray[0]=(node_t)-1;
	tree->levelArray[0]=0;
	for (node_t node=0; node < tree->num_nodes; node+=1){
		tree->vertexArray[node]=edge_pointer;
		tree->descendantArray[node]=1;
		tree->descendantArray_rec[node]=1;
		tree->descendantArray_gpu[node]=1;
		tree->descendantArray_gpu_np[node]=1;
		tree->descendantArray_gpu_np_hier[node]=1;
		//create the children
		for (int i=0;i< child_nodes[node];i++){
			tree->edgeArray[edge_pointer++]=++last_created;
			tree->parentArray[last_created]=node;	
			tree->levelArray[last_created]=tree->levelArray[node]+1;
		}
				
	}
	tree->vertexArray[tree->num_nodes]=edge_pointer;
	delete [] child_nodes;
} 



void tree_to_dot(tree_t tree, FILE *file){
	if (tree.num_nodes==0){
		printf("empty tree!\n");
		return;
	}
	else{
		fprintf(file, "digraph tree {\n");
		for (unsigned node=0; node<tree.num_nodes; node++){
			fprintf(file, " %llu [shape=circle,label=\"%llu",node,node);
                        fprintf(file,"-%llu\"];\n",tree.descendantArray[node]);
		
			for (unsigned edge=tree.vertexArray[node]; edge<tree.vertexArray[node+1];edge++){
				fprintf(file, "%llu -> %llu;\n", node, tree.edgeArray[edge]);
			}
		}
		fprintf(file, "}\n");
			
	}
}

void descendants_rec(tree_t *tree, node_t node){
	if (tree->vertexArray[node]!=tree->vertexArray[node+1]){
		for(node_t edge=tree->vertexArray[node]; edge<tree->vertexArray[node+1];edge++){
			node_t child = tree->edgeArray[edge];
			descendants_rec(tree,child);
			tree->descendantArray_rec[node]+=tree->descendantArray_rec[child];
		}
	}
}


void descendants_rec(tree_t *tree){
	descendants_rec(tree, 0);
}

void descendants(tree_t *tree){
	for (node_t node=0; node<tree->num_nodes; node++){
		for (node_t parent = tree->parentArray[node]; parent != (node_t) -1; parent = tree->parentArray[parent]){
			tree->descendantArray[parent]+=1;
		}
	}
}
