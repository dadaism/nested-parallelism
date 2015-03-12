/*
 * Copyright (c) 2007 Michela Becchi and Washington University in St. Louis.
 * All rights reserved
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    3. The name of the author or Washington University may not be used
 *       to endorse or promote products derived from this source code
 *       without specific prior written permission.
 *    4. Conditions of any other entities that contributed to this are also
 *       met. If a copyright notice is present from another entity, it must
 *       be maintained in redistributions of the source code.
 *
 * THIS INTELLECTUAL PROPERTY (WHICH MAY INCLUDE BUT IS NOT LIMITED TO SOFTWARE,
 * FIRMWARE, VHDL, etc) IS PROVIDED BY  THE AUTHOR AND WASHINGTON UNIVERSITY
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR WASHINGTON UNIVERSITY
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS INTELLECTUAL PROPERTY, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * File:   wgraph.h
 * Author: Michela Becchi, Thomas Nabelek
 * Email:  mbecchi@cse.wustl.edu
 * Organization: Applied Research Laboratory
 * 
 * Description: Implements an undirected graph with weighted edges.
 * 
 */

#ifndef __WGRAPH_H
#define __WGRAPH_H

typedef unsigned int vertex;
typedef unsigned int edge;
typedef unsigned int weight;

enum graph_type_
  {
    proper,
    multigraph
  };

typedef struct _scale_free_info { // For scale_free network generation reporting
  int degree;
  int expected_num_vertices_with_degree;
  int num_vertices_with_degree;
  float percentage;
  _scale_free_info *next;
} scale_free_info; 

class wgraph {
  unsigned int N;     // max number of vertices 
  unsigned long M;        //max number of edges

  struct wgedge {
    vertex l, r;  // endpoints of the edge
    weight wt;  // edge weight
    weight label;  // edge label
    edge lnext;  // link to next edge incident to l
    edge rnext;  // link to next edge incident to r
    bool joined;  // For small_world_network - don't rewire edge if it wasn't joined in the first place
  } *edges;

  typedef struct _path_member { // For shortest_path linked list and clustering coefficeint linked list
    vertex vertex_number;
    _path_member *next_vertex;
  } path_member;

  typedef struct __shortest_path_info { // For shortest_path linked list
    _path_member *head;
    int path_length;
  } _shortest_path_info;

  edge *firstedge;    // firstedge[v] is first edge at v
  scale_free_info *scale_free_info_head; // For scale-free network generation reporting
  scale_free_info *scale_free_info_tail;
  int *adjacencies;
  int *path_lengths;
  int *temp_matrix;
  int *vertex_levels; // For BFS
  _shortest_path_info *shortest_path_info;
  int *tentative_distances; // For shortest_path, characteristic_path_length

  void  hsort();    // heap sort subroutine
  void  bldadj();   // build adjacency lists
  bool  getedge(edge,FILE*);  // get edge from file
  char  cflush(char, FILE*);
  public:
  wgraph(unsigned int=100,unsigned long =1000);
  wgraph(const wgraph *);
  ~wgraph();
  unsigned int n;     // number of vertices
  unsigned long m;      // number of edges
  edge  first(vertex);    // return first edge incident to v
  edge  next(vertex,edge);  // return next edge of v, following e
  vertex  left(edge);   // return left endpoint of e
  vertex  right(edge);    // return right endpoint of e
  vertex  mate(vertex,edge);  // return other endpoint of e
  weight  w(edge);    // return weight of e
  weight  label(edge);    // return label of e
  void  changeWt(edge,weight);  // change weight of e
  edge  join(vertex,vertex,weight); // join two vertices with an edge
  // bool  get(FILE*);   // get graph from file
  // void  put(FILE*);   // put graph to file
  void  esort();    // sort edges by weight
  void  reset();        //reset the graph
  void directed_to_dot(char *filename);
  void undirected_to_dot(char *filename);
  void save_gr(FILE *, int, int, int, int, int, bool, graph_type_);  // Save random graph to file
  void save_gr(FILE *, int, int, int, int, bool, double, int, graph_type_);  // Save small-world graph to file
  void save_gr(FILE *, int, int, int, double, bool);  // Save scale-free graph to file
  void random_graph_undirected(int, int, int, int, graph_type_);
  void random_graph_directed(int, int, int, int, graph_type_);
  void random_graph_undirected_old1(int, int, int, int, graph_type_);
  void bld_adj_mat(int, int, bool);
  void bld_adj_mat_rewire(int, int);
  void print_adj_mat();
  void path_length(int);
  void print_length_matrix();
  void bfs_parallel(vertex = 0);
  void bfs_report(vertex = 0);
  bool bfs_connect_check(vertex = 0);
  void small_world_lattice(int, int, int);
  void small_world_randomize(int, int, double, bool, graph_type_);
  scale_free_info *scale_free(int, int, double);
  void shortest_path(int, vertex, vertex);
  void shortest_path_report();
  void print_shortest_path_matrix();
  double characteristic_path_length(int);
  double clustering_coefficient(int, int);
  
  unsigned int joined_edge_count; // Keep track of number of joined edges
  unsigned int rewired_edge_count; // Keep track of number of rewired edges (small world)
};

// Return first edge incident to x.
inline edge wgraph::first(vertex v) { return firstedge[v]; };

// Return next edge incident to v, following e.
inline edge wgraph::next(vertex v,edge e)
{ return edges[e].l == v ? edges[e].lnext : edges[e].rnext; }

// Return left endpoint of e.
inline vertex wgraph::left(edge e) { return edges[e].l; }

// Return right endpoint of e.
inline vertex wgraph::right(edge e) { return edges[e].r; }

// Return other endpoint of e.
inline vertex wgraph::mate(vertex v, edge e)
{ return edges[e].l == v ? edges[e].r : edges[e].l ; }

// Return weight of e.
inline weight wgraph::w(edge e) { return edges[e].wt; }

// Return label of e.
inline weight wgraph::label(edge e) { return edges[e].label; }

// Change weight of e.
inline void wgraph::changeWt(edge e, weight w) { edges[e].wt = w; }

#endif /*__WGRAPH_H */
