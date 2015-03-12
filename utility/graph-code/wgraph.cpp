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
 * */

/*
 * File:   wgraph.cpp
 * Author: Jon Turner, Michela Becchi, Thomas Nabelek
 * Email:  mbecchi@cse.wustl.edu
 * Organization: Applied Research Laboratory
 * Comments:  adapted from Jon Turner's code, class CSE542: http://www.arl.wustl.edu/~jst/cse/542/src/index.html
 * 
 */

#include <limits.h>
#include <math.h> // scale_free() uses pow() and floor() or round()

wgraph::wgraph(unsigned int N1, unsigned long M1) { // N1 is num_vertices, M1 is max_num_edges
  N = N1;
  M = M1;
  n = N;
  m = 0;

  firstedge = new edge[N+1];
  edges = new wgedge[M+1];
  for (vertex u = 0; u <= n; u++)
    firstedge[u] = 0;
  for (edge u = 0; u <= M; u++) {
    edges[u].l = edges[u].r = edges[u].lnext = edges[u].rnext = 0;
    edges[u].joined = false;
  }
  edges[0].wt = 0;

  adjacencies = new int[n*n]; // For path length calculations and creation of proper graphs
  for (int i = 0; i < n*n; i++)
    adjacencies[i] = 0;

  vertex_levels = new int[n]; // For BFS level check
  joined_edge_count = 0; // Keep track of number of joined edges
  rewired_edge_count = 0; // Keep track of rewired edges (for small world)

  shortest_path_info = new _shortest_path_info; // For shortest_path
  tentative_distances = new int[n]; // For shortest_path, characteristic_path_length
}

// Copy constructor
wgraph::wgraph(const wgraph *original) {
  N = original->N;
  M = original->M;
  n = original->n;
  m = original->m;

  firstedge = new edge[N+1];
  for (int i = 0; i < N+1; i++)
    firstedge[i] = original->firstedge[i];
  edges = new wgedge[M+1];
  for (int i = 0; i < M+1; i++)
    edges[i] = original->edges[i];

  adjacencies = new int[n*n]; // For path length calculations and creation of proper graphs
  for (int i = 0; i < n*n; i++)
    adjacencies[i] = original->adjacencies[i];

  vertex_levels = new int[n]; // For BFS level check
  joined_edge_count = original->joined_edge_count; // Keep track of number of joined edges
  rewired_edge_count = original->rewired_edge_count; // Keep track of rewired edges (for small world)

  shortest_path_info = new _shortest_path_info; // For shortest_path
  tentative_distances = new int[n]; // For shortest_path, characteristic_path_length
}

wgraph::~wgraph() {
  delete [] firstedge;
  delete [] edges;
  delete [] adjacencies;
  delete [] vertex_levels;
  delete [] tentative_distances;

  // Deallocate predecessor path linked list
  path_member* previous = shortest_path_info->head;
  path_member* current = shortest_path_info->head;
  while (previous != NULL)
    {
      current = previous->next_vertex;
      free(previous);
      previous = current;
    }

  delete shortest_path_info;
}

// Disabled -- not updated from original code
// void wgraph::reset() {
//  delete [] firstedge; 
//  delete [] edges;
//  firstedge = new edge[N+1];
//  edges = new wgedge[M+1];
//  n = N;
//  m = 0;
//  for (vertex u = 0; u <= n; u++) firstedge[u] = Null;
//  for (edge u = 0; u <= M; u++) edges[u].l = edges[u].r = edges[u].lnext = edges[u].rnext = 0;
//  edges[0].wt = 0;
  
//  delete [] adjacencies;
//  adjacencies = new int[n*n]; // For path length calculations
//  for (int i = 0; i < N; i++)
//    for (int j = 0; j < N; j++)
//      adjacencies[i*n+j] = 0;

//  delete [] vertex_levels;
//  delete shortest_path_info;
// }


char wgraph::cflush(char c, FILE *f) {
// Read up to first occurrence of c or EOF.
  char c1; while ((c1 = getc(f)) != EOF && c1 != c) {} return c1;
}

// Read one edge from *f into edges[e]. Return true on success, else false.
bool wgraph::getedge(edge e, FILE* f) {
  char c;
  if (cflush('(',f) == EOF) return false;
  
  fscanf(f,"%d",&edges[e].l);
  if (cflush(',',f) == EOF) return false;
  fscanf(f,"%d",&edges[e].r);
  
  if (cflush(',',f) == EOF) return false;
  fscanf(f,"%d",&edges[e].wt);
  if (cflush(')',f) == EOF) return false;
  return true;
}

// Join u and v with edge of given weight. Return edge number.
edge wgraph::join(vertex u, vertex v, weight W) {
  if (++m > M)
    fatal("wgraph: too many edges");
  if (u > N || v > N)
    fatal("wgraph: left and/or right vertex number exceeds number of vertices");
  
  edges[m].l = u; edges[m].r = v; edges[m].wt = W; edges[m].label = m;
  edges[m].lnext = firstedge[u];
  firstedge[u] = m;
  edges[m].rnext = firstedge[v];
  firstedge[v] = m;
  edges[m].joined = true; // For small_world_network - don't rewire edge if it wasn't joined in the first place
  joined_edge_count++; // Keep track of number of joined edges
  return m;
}

// Build adjacency lists.
void wgraph::bldadj() {
  for (vertex u = 0; u < n; u++) firstedge[u] = Null;
  for (edge e = m; e >= 1; e--) {
    edges[e].lnext = firstedge[edges[e].l];
    firstedge[edges[e].l] = e;
    edges[e].rnext = firstedge[edges[e].r];
    firstedge[edges[e].r] = e;
  }
}

// // Get graph from f.
// bool wgraph::get(FILE* f) {
//  if (fscanf(f,"%d%ld",&N,&M) != 2)
//    return false;

//  delete [] firstedge;
//  delete [] edges;
//  firstedge = new edge[N+1];
//  edges = new wgedge[M+2];

//  n = N; m = 1;
//  for (int i = 1; i <= 2*M; i++) {
//    // each edge appears twice in input
//    if (!getedge(m,f))
//      break;
//    if (edges[m].l > n || edges[m].r > n)
//      fatal("wgraph::get: out of range vertex number");
//    if (edges[m].l < edges[m].r) {
//      if (m > M) fatal("wgraph::get: edge mismatch");
//    m++;
//    }
//  }
//  m--; bldadj();
//  return true;
// }

// // Put graph out on f.
// void wgraph::put(FILE* f) {
//  int i; vertex u; edge e;
//  fprintf(f,"%d %ld\n",n,m);
//  for (u = 0; u < n; u++) {
//    i = 0;
//    for (e = first(u); e != Null; e = next(u,e)) {
//      fprintf(f,"%u=(%2d,%2d,%2d)  ",e, u,mate(u,e),w(e));
//             if ((++i)%5 == 0) putc('\n',f);
//         }
//  }
//  putc('\n',f);
// }

// Sort the edges by cost.
void wgraph::esort() {
  hsort(); bldadj();
}

// Sort edges according to weight, using heap-sort.
void wgraph::hsort() {
  int i, mm, p, c; weight w; wgedge e;

  for (i = m/2; i >= 1; i--) {
    // do pushdown starting at i
    e = edges[i]; w = e.wt; p = i;
    while (1) {
      c = 2*p;
      if (c > m) break;
      if (c+1 <= m && edges[c+1].wt >= edges[c].wt) c++;
      if (edges[c].wt <= w) break;
      edges[p] = edges[c]; p = c;
    }   
    edges[p] = e;
  }
  // now edges are in heap-order with largest weight edge on top

  for (mm = m-1; mm >= 1; mm--) {
    e = edges[mm+1]; edges[mm+1] = edges[1];
    // now largest edges are m, m-1,..., mm+1
    // edges 1,...,mm form a heap with largest weight edge on top
    // pushdown from 1
    w = e.wt; p = 1;
    while (1) {
      c = 2*p;
      if (c > mm) break;
      if (c+1 <= mm && edges[c+1].wt >= edges[c].wt) c++;
      if (edges[c].wt <= w) break;
      edges[p] = edges[c]; p = c;
    }   
    edges[p] = e;
  }
}

void wgraph::directed_to_dot(char *filename) {
  FILE *file=fopen(filename,"w");
  fprintf(file, "digraph \"%s\" {\n", filename);
  for (vertex u = 0; u < n; u++) {
    fprintf(file, " %d [shape=circle,label=\"%d\"];\n",u,u);
  }
  for (edge e = 1; e <=m; e++) {
      fprintf(file, "%d -> %d [shape=none,label=\"%u\"];\n", left(e),right(e),w(e));
  }  
  fprintf(file, "}\n");
  fclose(file);
}

void wgraph::undirected_to_dot(char *filename) {
  FILE *file=fopen(filename,"w");
  fprintf(file, "graph \"%s\" {\ngraph [ dpi = 150, bgcolor=\"transparent\"];\n", filename);
  for (vertex u = 0; u < n; u++) {
    fprintf(file, " %d [shape=circle,label=\"%d\",width=.4,fixedsize=true];\n",u,u);
  }
  for (edge e = 1; e <=m; e++) {
    fprintf(file, " %d -- %d [shape=none, fontsize=10, label=\"%u (%u)\"];\n", left(e),right(e),w(e), e);
  }

  // fprintf(file, "legend [shape=rectangle,label=\"test\",rank=\"max\"]\n"); // Create legend -- may make data invalid for reading back in

  fprintf(file, "}\n");
  fclose(file);
}

// Save random graph to file
void wgraph::save_gr(FILE *data_file, int num_vertices, int num_edges, int min_degree, int max_degree, int max_weight, bool directed_graph, graph_type_ graph_type)
  {
    // Print graph parameters
    fprintf(data_file, "# Generic random connected ");
    directed_graph ? fprintf(data_file, "directed ") : fprintf(data_file, "undirected ");
    (graph_type == proper)  ? fprintf(data_file, "proper graph ") : fprintf(data_file, "multigraph ");
    fprintf(data_file, "\n#\n# Nodes: %d Edges: %d\n# FromNodeID  ToNodeID", num_vertices, num_edges);

    // Print edges
    for (edge e = 1; e <= joined_edge_count; e++)
      fprintf(data_file, "\n%u %u", left(e), right(e));
  }

// Save small-world graph to file
void wgraph::save_gr(FILE *data_file, int num_vertices, int num_edges, int initial_degree, int max_weight, bool directed_graph, double probability, int rewird_count, graph_type_ graph_type)
  {
    // Print graph parameters
    fprintf(data_file, "# Small-world connected ");
    directed_graph ? fprintf(data_file, "directed ") : fprintf(data_file, "undirected ");
    (graph_type == proper)  ? fprintf(data_file, "proper graph ") : fprintf(data_file, "multigraph ");
    fprintf(data_file, "\n# probability: %lf, number of rewird edges: %d (%f%%)", probability, rewird_count, ((float)rewird_count/(float)num_edges)*100);
    fprintf(data_file, "\n# Nodes: %d Edges: %d\n# FromNodeID  ToNodeID", num_vertices, num_edges);

    // Print edges
    for (edge e = 1; e <= joined_edge_count; e++)
      fprintf(data_file, "\n%u %u", left(e), right(e));
  }

// Save scale-free graph to file
void wgraph::save_gr(FILE *data_file, int num_vertices, int num_edges, int max_weight, double gamma, bool directed_graph)
  {
    // Print graph parameters
    fprintf(data_file, "# Scale-free connected ");
    directed_graph ? fprintf(data_file, "directed ") : fprintf(data_file, "undirected ");
    fprintf(data_file, "proper graph");
    fprintf(data_file, "\n# gamma: %lf", gamma);
    fprintf(data_file, "\n# Nodes: %d Edges: %d\n# FromNodeID  ToNodeID", num_vertices, num_edges);

    // Print edges
    for (edge e = 1; e <= joined_edge_count; e++)
      fprintf(data_file, "\n%u %u", left(e), right(e));
  }

// random_graph_connected first builds a spanning tree with the appropriate number of vertices and then fills in the tree with edges to give the vertices higher degree values
// void wgraph::random_graph_undirected_old1(int num_vertices, int min_degree, int max_degree, int max_weight, graph_type_ graph_type)
//   {
//     srand(time(NULL));

//     int vertex_degrees[num_vertices];
//     for (int i = 0; i < num_vertices; i++)
//       vertex_degrees[i] = (rand() % (max_degree - min_degree + 1)) + min_degree;

//     for (int i = 0; i < num_vertices; i++)
//       printf("\n%d : %d", i, vertex_degrees[i]);

//     int degree_sum = 0;
//     for (int j = 0; j < num_vertices; j++) // Find out the sum of the degrees remaining
//       degree_sum += vertex_degrees[j];

//     bool in_graph[num_vertices];
//     for (int i = 0; i < num_vertices; i++)
//       in_graph[i] = false;

//     int left, right;
//     bool need_right, need_left;
//     int remaining_vertices = num_vertices;

//     // -------- BUILD SPANNING TREE --------
//     in_graph[rand() % num_vertices] = true;
//     remaining_vertices--;
//     while (remaining_vertices > 0)
//       {
//         need_left = need_right = true;
//         while (need_left)
//           {
//             left = rand() % num_vertices;
//             if (in_graph[left] && vertex_degrees[left] > 0) // Choose left vertex from vertices already in graph
//               need_left = false;
//           }
//         while (need_right)
//           {
//             right = rand() % num_vertices;
//             if ( !in_graph[right] && (vertex_degrees[right] > 0) ) // Choose right vertex from vertices not in graph
//               need_right = false;
//           }
//         this->join(left, right, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
//         this->bld_adj_mat(left, right);
//         in_graph[right] = true;
//         vertex_degrees[left]--;
//         vertex_degrees[right]--;
//         degree_sum -= 2;
//         remaining_vertices--; // count right vertex as added to spanning tree
//       }

//       // Remove a degree of 1 from some vertices to allow for leaves (necessary because required degree of at least 2 to build tree above)
//       for (int i = 0; i < num_vertices; i++)
//         {
//           if ( (i%3 == 0) && (vertex_degrees[i] != 0) )
//             vertex_degrees[i]--;
//         }

//     // -------- FILL IN REMAINING EDGES --------
//     // try_count and ok_to_join are necessary becasue without them, while (need_right) may get stuck in a loop
//     if (degree_sum >= 2) // If sum of degrees remaining is less than 2, don't execute loop (if less than 2, there are no neighbors left)
//       {
//         int try_count;
//         bool ok_to_join;
//         for (int i = 0; i < num_vertices; i++)
//           {
//             while (vertex_degrees[i] > 0)
//               {
//                 left = i;
//                 need_right = true;
//                 try_count = 0;
//                 ok_to_join = true;

//                 while (need_right && try_count < num_vertices*2)
//                   {
//                     right = rand() % num_vertices;

//                     if (graph_type == proper) // If proper graph, do not allow self loops or multiple edges between same two vertices; pick different right vertex
//                       while ( (left == right) || (adjacencies[left*n + right] >= 1) )
//                         right = rand() % num_vertices;

//                     if (left == right) // Only for multigraphs
//                       {
//                         if (vertex_degrees[right] > 1)
//                           need_right = false;
//                       }
//                     else
//                       {
//                         if (vertex_degrees[right] > 0)
//                           need_right = false;
//                       }

//                     try_count++;
//                     if (try_count == num_vertices*2) // After a lot of tries, if no right neighbor was found, do not join and move on
//                       ok_to_join = false;
//                   }
//                 if (ok_to_join) // Only join if a right neighbor was found
//                   {
//                     this->join(left, right, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
//                     this->bld_adj_mat(left, right);
//                     vertex_degrees[left]--;
//                     vertex_degrees[right]--;
//                     degree_sum -= 2;
//                   }

//                 if (degree_sum < 2 || !ok_to_join) // If sum of degrees remaining is less than 2, exit loop (if less than 2, there are no neighbors left)
//                   {
//                     i = num_vertices;
//                     vertex_degrees[i] = 0;
//                   }
//               }
//           }
//       }
//   }

// random_graph_connected first builds a spanning tree with the appropriate number of vertices and then fills in the tree with edges to give the vertices higher degree values
void wgraph::random_graph_undirected(int num_vertices, int min_degree, int max_degree, int max_weight, graph_type_ graph_type)
  {
    srand(time(NULL));

    int vertex_degrees[num_vertices];
    bool in_graph[num_vertices];
    for (int i = 0; i < num_vertices; i++)
      {
        vertex_degrees[i] = 0;
        in_graph[i] = false;
      }

    int left, right;
    bool need_right, need_left;
    int remaining_vertices = num_vertices;

    // -------- BUILD SPANNING TREE --------
    in_graph[rand() % num_vertices] = true;
    remaining_vertices--;
    while (remaining_vertices > 0)
      {
        need_left = need_right = true;
        while (need_left)
          {
            left = rand() % num_vertices;
            if (in_graph[left] && vertex_degrees[left] < max_degree) // Choose left vertex from vertices already in graph
              need_left = false;
          }
        while (need_right)
          {
            right = rand() % num_vertices;
            if ( !in_graph[right] && (vertex_degrees[right] < max_degree) ) // Choose right vertex from vertices not in graph
              need_right = false;
          }
        this->join(left, right, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
        this->bld_adj_mat(left, right, false);
        in_graph[right] = true;
        vertex_degrees[left]++;
        vertex_degrees[right]++;
        remaining_vertices--; // count right vertex as added to spanning tree
      }

    for (int i = 0; i < num_vertices; i++)
      vertex_degrees[i] = ((rand() % (max_degree - min_degree + 1)) + min_degree) - vertex_degrees[i];

    int degree_sum = 0;
    for (int j = 0; j < num_vertices; j++) // Find out the sum of the degrees remaining
      degree_sum += vertex_degrees[j];

    // -------- FILL IN REMAINING EDGES --------
    // try_count and ok_to_join are necessary becasue without them, while (need_right) may get stuck in a loop
    if (degree_sum >= 2) // If sum of degrees remaining is less than 2, don't execute loop (if less than 2, there are no neighbors left)
      {
        int try_count;
        bool ok_to_join;
        for (int i = 0; i < num_vertices; i++)
          {
            while (vertex_degrees[i] > 0)
              {
                left = i;
                need_right = true;
                try_count = 0;
                ok_to_join = true;

                while (need_right && try_count < num_vertices*2)
                  {
                    right = rand() % num_vertices;

                    if (graph_type == proper) // If proper graph, do not allow self loops or multiple edges between same two vertices; pick different right vertex
                      while ( (left == right) || (adjacencies[left*n + right] >= 1) )
                        right = rand() % num_vertices;

                    if (left == right) // Only for multigraphs
                      {
                        if (vertex_degrees[right] > 1)
                          need_right = false;
                      }
                    else
                      {
                        if (vertex_degrees[right] > 0)
                          need_right = false;
                      }

                    try_count++;
                    if (try_count == num_vertices*2) // After a lot of tries, if no right neighbor was found, do not join and move on
                      ok_to_join = false;
                  }
                if (ok_to_join) // Only join if a right neighbor was found
                  {
                    this->join(left, right, (rand() % max_weight) + 1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
                    this->bld_adj_mat(left, right, false);
                    vertex_degrees[left]--;
                    vertex_degrees[right]--;
                    degree_sum -= 2;
                  }

                if (degree_sum < 2 || !ok_to_join) // If sum of degrees remaining is less than 2, exit loop (if less than 2, there are no neighbors left)
                  {
                    i = num_vertices;
                    vertex_degrees[i] = 0;
                  }
              }
          }
      }
  }

// random_graph_connected first builds a spanning tree with the appropriate number of vertices and then fills in the tree with edges to give the vertices higher degree values
void wgraph::random_graph_directed(int num_vertices, int min_degree, int max_degree, int max_weight, graph_type_ graph_type)
  {
    srand(time(NULL));

    int vertex_degrees[num_vertices];
    bool in_graph[num_vertices];
    for (int i = 0; i < num_vertices; i++)
      {
        vertex_degrees[i] = 0;
        in_graph[i] = false;
      }

    int left, right;
    bool need_left, need_right;
    int remaining_vertices = num_vertices;

    // -------- BUILD SPANNING TREE --------
    in_graph[rand() % num_vertices] = true;
    remaining_vertices--;
    while (remaining_vertices > 0)
      {
        need_left = true;
        need_right = true;
        while (need_left)
          {
            left = rand() % num_vertices;

            if (in_graph[left] && vertex_degrees[left] < max_degree) // Choose left vertex from vertices already in graph
              need_left = false;
          }
        while (need_right)
          {
            right = rand() % num_vertices;
            if (!in_graph[right]) // Choose right vertex from vertices not in graph
              need_right = false;
          }

        this->join(left, right, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
        this->bld_adj_mat(left, right, true);
        in_graph[right] = true;
        vertex_degrees[left]++;
        remaining_vertices--; // count right vertex as added to spanning tree
      }

    for (int i = 0; i < num_vertices; i++)
      vertex_degrees[i] = ((rand() % (max_degree - min_degree + 1)) + min_degree) - vertex_degrees[i];
    
    int degree_sum = 0;
    for (int j = 0; j < num_vertices; j++) // Find out the sum of the degrees remaining
      degree_sum += vertex_degrees[j];

    // -------- FILL IN REMAINING EDGES --------
    if (degree_sum >= 1) // If sum of degrees remaining is less than 2, don't execute loop (if less than 2, there are no neighbors left)
      {
        for (int i = 0; i < num_vertices; i++)
          {

            while (vertex_degrees[i] > 0)
              {
                left = i;
                right = rand() % num_vertices;

                if (graph_type == proper) // If proper graph, multiple edges having the same orientation between same two vertices; pick different right vertex
                  while (adjacencies[left*n + right] >= 1) 
                    right = rand() % num_vertices;

                this->join(left, right, (rand() % max_weight) + 1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
                this->bld_adj_mat(left, right, true);
                vertex_degrees[left]--;
                degree_sum--;

                if (degree_sum < 1) // If sum of degrees remaining is less than 1, exit loop
                  {
                    i = num_vertices;
                    vertex_degrees[i] = 0;
                  }
              }
          }
      }
  }

// Create small-world lattice given number of vertices and degree of each vertex (degree must be even - enforced in main())
void wgraph::small_world_lattice(int num_vertices, int degree, int max_weight)
  {
    // srand(time(NULL));

    int left, right;
    int connect_distance;
    for (int i = 2; i <= degree; i += 2)
      {
        connect_distance = i/2;
        for (left = 0; left < num_vertices; left++)
          {
            if (left == 0)
              right = connect_distance;
            else if (right == num_vertices-1)
              right = 0;
            else
              right++;
            // this->join(left, right, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
            this->join(left, right, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
            this->bld_adj_mat(left, right, false); // Add new connection to adjacency matrix
          }
      }
  }

// Randomize small-world lattice given probability (must be between 0 and 1 - enforced in main())
void wgraph::small_world_randomize(int num_vertices, int num_edges, double probability, bool directed, graph_type_ graph_type)
  {
    vertex original_right, new_right;
    int rewire_attempt;
    for (int i = 1; i <= num_edges; i++)
      {
        if (edges[i].joined) // Don't rewire edge if it wasn't joined in the first place
          {
            if ((double)rand()/(double)RAND_MAX < probability) // If random number between 0 and 1 is less than given probability, rewire edge
              {
                // Assign new right vertexat random
                original_right = edges[i].r;
                new_right = rand() % num_vertices;
                rewire_attempt = 0;

                if (graph_type == proper) // Disallow self-loops and multple edges between two vertices for undirected graphs
                  while( (new_right == edges[i].l) || (adjacencies[edges[i].l*num_vertices + new_right] > 0)  && rewire_attempt++ < num_vertices*num_vertices)
                    new_right = rand() % num_vertices;

                if (rewire_attempt == num_vertices*num_vertices+1) // If edge can't be rewired, continue to next edge.
                 continue;

                // Maintain edge link list
                if (first(original_right) != i)
                  {
                    edge previous_edge = -1;
                    for (int j = 0; j <= num_edges; j++)
                      {
                        if ((edges[j].r == original_right && edges[j].rnext == i) || (edges[j].l == original_right && edges[j].lnext == i))
                          {
                            previous_edge = j;
                            j = num_edges + 1; // Break out of loop
                          }
                      }
                    if (edges[previous_edge].r == original_right && edges[i].r == original_right)
                      edges[previous_edge].rnext = edges[i].rnext;
                    else if (edges[previous_edge].r == original_right && edges[i].l == original_right)
                      edges[previous_edge].rnext = edges[i].lnext;
                    else if (edges[previous_edge].l == original_right && edges[i].r == original_right)
                      edges[previous_edge].lnext = edges[i].rnext;
                    else if (edges[previous_edge].l == original_right && edges[i].l == original_right)
                      edges[previous_edge].lnext = edges[i].lnext;
                  }
                else // if (first(original_right) == i)
                  firstedge[original_right] = edges[i].rnext;
                edges[i].rnext = firstedge[new_right];
                firstedge[new_right] = i;
                edges[i].r = new_right;

                this->bld_adj_mat_rewire(edges[i].l, original_right); // Remove original connection from adjacency matrix
                this->bld_adj_mat(edges[i].l, new_right, false);  // Add new connection to adjacency matrix

                rewired_edge_count++; // Keep track of rewired edges (for small world)
              }
          }
        else // Break out of loop if an unjoined edge is processed, as edges are joined in numerical order
          i = num_edges+1;
      }
    // this->print_adj_mat();
  }

// Create scale-free network given number of vertices, max_weight of each edge, and gamma value
scale_free_info *wgraph::scale_free(int num_vertices, int max_weight, double gamma)
  {
    // ----- Find number of vertices for each degree -----
    int degree, highest_degree;
    vertex k, highest_degree_vertex;
    double num_vertices_with_degree[2*num_vertices+1];
    double vertex_sum = 0;
    
    bool scale_values = false;

    for (degree = 1; degree < 2*num_vertices+1; degree++)
      num_vertices_with_degree[degree] = scale_values ? (num_vertices/(pow(degree, gamma))) : round(num_vertices/(pow(degree, gamma)));

    for (degree = 1; degree < 2*num_vertices+1; degree++)
      vertex_sum += num_vertices_with_degree[degree];

    for (degree = 1; degree < 2*num_vertices+1; degree++)
      {
        if (scale_values)
          num_vertices_with_degree[degree] = round(((double)num_vertices/(double)vertex_sum)*num_vertices_with_degree[degree]);
        if (num_vertices_with_degree[degree] == 0)
          {
            highest_degree = degree - 1;
            if (highest_degree == 1) // Highest degree cannot be 1
              highest_degree = 2;
            break;
          }
      }

    if (scale_values)
      {
        vertex_sum = 0;
        for (degree = 1; degree < 2*num_vertices+1; degree++)
          vertex_sum += num_vertices_with_degree[degree];
      }

    if (vertex_sum > num_vertices)
      num_vertices_with_degree[1] = num_vertices_with_degree[1] - (vertex_sum - num_vertices);

    // // Print number of degrees for each vertex and sum
    // for (degree = 1; degree < num_vertices/3; degree++)
    //  printf("\ndegree %2d: %2d vertices", degree, (int)num_vertices_with_degree[degree]);

    // ----- Assign vertex degrees -----
    int vertex_degrees_original[num_vertices];
    for (k = 0; k < num_vertices; k++)
      vertex_degrees_original[k] = -1;

    degree = 3; // Start with degree 3 (all vertices must have degree of at least 2 to build spanning tree - degree 2 assigned below)
    int remaining_num_for_degree = num_vertices_with_degree[degree];
    k = rand() % num_vertices;
    for (int i = 0; i < num_vertices; i++)
      {
        if (remaining_num_for_degree == 0)
          {
            degree++;
            remaining_num_for_degree = num_vertices_with_degree[degree];
          }
        if (remaining_num_for_degree != 0)
          {
            while (vertex_degrees_original[k] != -1) // Assign degrees to radomly selected vertices
              k = rand() % num_vertices;
            vertex_degrees_original[k] = degree;

            if (remaining_num_for_degree != 0)
              highest_degree_vertex = k;

            remaining_num_for_degree--;
          }
      }

    for (k = 0; k < num_vertices; k++) // Assign any vertices not already assigned a degree a degree of 2 (all vertices must have degree of at least 2 to build spanning tree)
      if (vertex_degrees_original[k] == -1)
        vertex_degrees_original[k] = 2;

    // ----- Build spanning tree -----
    int vertex_degrees_final[num_vertices];
    int remaining_degree_for_vertex[num_vertices];
    int in_graph[num_vertices];
    for (k = 0; k < num_vertices; k++)
      {
        vertex_degrees_final[k] = 0;
        in_graph[k] = false;
        remaining_degree_for_vertex[k] = vertex_degrees_original[k];
      }

    vertex left, right;
    bool need_right;
    int num_in_graph = 0;
    in_graph[rand() % num_vertices] = true; // Seed graph with random vertex
    num_in_graph++;

    while (num_in_graph < num_vertices)
      {
        left = rand() % num_vertices;
        if (in_graph[left])
          {
            for (right = 0; right < num_vertices; right++)
              if (!in_graph[right] && left != right && remaining_degree_for_vertex[right] > 0 && remaining_degree_for_vertex[left] > 0 && !adjacencies[left*n + right])
                {
                  this->join(left, right, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
                  this->bld_adj_mat(left, right, false);
                  vertex_degrees_final[left]++;
                  vertex_degrees_final[right]++;
                  remaining_degree_for_vertex[left]--;
                  remaining_degree_for_vertex[right]--;
                  in_graph[right] = true;
                  num_in_graph++;
                }

            // this->undirected_to_dot("mygraph_generated_scale_free.dot");
            // system("dot -Tjpeg mygraph_generated_scale_free.dot > mygraph_generated_scale_free.jpeg -q"); // '-q' suppresses warning messages.
            // usleep(500000);
        }
      }
    // // Print degree for each vertex
    // for (k = 0; k < num_vertices; k++)
    //   if (remaining_degree_for_vertex[k] != 0)
    //     printf("\nvertex %2d: %2d %2d %2d", k, vertex_degrees_original[k], vertex_degrees_final[k], remaining_degree_for_vertex[k]);
    // printf("\n\n");

    // ----- Fill degrees for each vertex
    for (k = 0; k < num_vertices; k++) // Set all degree 2 vertices that have not been filled to degree 1
      if (vertex_degrees_original[k] == 2 && remaining_degree_for_vertex[k] == 1)
        {
          vertex_degrees_original[k] = 1;
          remaining_degree_for_vertex[k] = 0;
        }

    // // Print degree for each vertex
    // for (k = 0; k < num_vertices; k++)
    //   if (remaining_degree_for_vertex[k] != 0)
    //     printf("\nvertex %2d: %2d %2d %2d", k, vertex_degrees_original[k], vertex_degrees_final[k], remaining_degree_for_vertex[k]);
    // printf("\n\n");


    int remaining_degree_sum = 0;
    for (k = 0; k < num_vertices; k++)
      remaining_degree_sum += remaining_degree_for_vertex[k];

    int try_count = 0;
    while (remaining_degree_sum > 1)
      {
        int right2; // necessary for for loop since right vertex is unsigned
        for (left = 0; left < num_vertices; left++)
          {
            if (remaining_degree_for_vertex[left] > 0)
              for (right2 = num_vertices-1; right2 >= 0; right2--)
                {
                  if (remaining_degree_for_vertex[left] > 0 && remaining_degree_for_vertex[right2] > 0 && right2 != left && !adjacencies[left*n + right]) // Must check remaining degree for left again
                    {
                      // printf("joining %d and %d\n", left, right2);
                      this->join(left, right2, (rand() % max_weight) +1); // If max_weight = 10, rand() % max_weight will return an int 0-9, so add 1 to get 1-10
                      this->bld_adj_mat(left, right2, false);
                      vertex_degrees_final[left]++;
                      vertex_degrees_final[right2]++;
                      remaining_degree_for_vertex[left]--;
                      remaining_degree_for_vertex[right2]--;
                      remaining_degree_sum -= 2;                  
                    }
                }
        try_count++;
          }

        if (try_count > num_vertices*num_vertices)
          break;
      }

    // // Print degree for each vertex
    // for (k = 0; k < num_vertices; k++)
    //   // if (remaining_degree_for_vertex[k] != 0)
    //     printf("\nvertex %2d: %2d %2d %2d", k, vertex_degrees_original[k], vertex_degrees_final[k], remaining_degree_for_vertex[k]);
    // printf("\nremaining sum is %d\n", remaining_degree_sum);

    // Print number of vertices for each degree
    int num_vertices_with_degree2;
    double expected_num_vertices_with_degree;

    scale_free_info_head = NULL;

    for (degree = 1; degree <= highest_degree; degree++)
      {
        num_vertices_with_degree2 = 0;
        if (scale_values)
          expected_num_vertices_with_degree = round(((double)num_vertices/(double)vertex_sum)*(num_vertices/(pow(degree, gamma))));
        else
          expected_num_vertices_with_degree = round(num_vertices/(pow(degree, gamma)));

        for (k = 0; k < num_vertices; k++)
          if (vertex_degrees_final[k] == degree)
            num_vertices_with_degree2++;

        if (!scale_free_info_head)
          {
            scale_free_info_head = new scale_free_info;
            scale_free_info_tail = scale_free_info_head;
          }
        else
          {
            scale_free_info_tail->next = new scale_free_info;
            scale_free_info_tail = scale_free_info_tail->next;
          }

        scale_free_info_tail->degree = degree;
        scale_free_info_tail->expected_num_vertices_with_degree = expected_num_vertices_with_degree;
        scale_free_info_tail->num_vertices_with_degree = num_vertices_with_degree2;
        scale_free_info_tail->percentage = (double)num_vertices_with_degree2/expected_num_vertices_with_degree;
      }
      
    return scale_free_info_head;
  }

void wgraph::bld_adj_mat(int left, int right, bool directed_graph)
  {
    adjacencies[left*n + right]++;
    if (left != right && !directed_graph) // Do not count loops twice (they are on the diagonal of the adjcency matrix, so symmetry is maintained); for directed graphs, only count left to right, not right to left
      adjacencies[right*n + left]++;
  }

// void wgraph::bld_adj_mat(int left, int right)
//   {
//     adjacencies[left*n + right]++;
//     if (left != right) // Do not count loops twice (they are on the diagonal of the adjcency matrix, so symmetry is maintained)
//       adjacencies[right*n + left]++;
//   }

void wgraph::bld_adj_mat_rewire(int left, int right)
  {
    adjacencies[left*n + right]--;
    if (left != right) // Do not count loops twice (they are on the diagonal of the adjcency matrix, so symmetry is maintained)
      adjacencies[right*n + left]--;
  }

void wgraph::print_adj_mat()
  {
    printf("\n\n ");
    for (int i = 0; i < n; i++)
      printf("%3d", i);
    for (int i = 0; i < n; i++)
      {
        printf("\n%d", i);
        for (int j = 0; j < n; j++)
          printf("%3d", adjacencies[i * n + j]);
      }
    printf("\n");
  }

void wgraph::path_length(int length)
  {
    path_lengths = new int[n*n]; // For path length calculations
    temp_matrix = new int[n*n]; // For path length calculations

    int i, j, k, l, m;

    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
        path_lengths[i*N+j] = adjacencies[i*N+j];


    for (k = 1; k < length; k++) // Repeat matrix multiplication the number of times given by the desired path length
      {
        int temp_sum = 0;

        int current_row, current_column;

        for (l = 0; l < N; l++) // For each element of the path_lengths matrix, do matrix multiplication
          for (m = 0; m < N; m++)
            {
              temp_sum = 0;
              for (i = 0; i < N; i++)
                temp_sum += path_lengths[l*N+i] * adjacencies[i*N+m];

              temp_matrix[l*N+m] = temp_sum;
            }

        for (i = 0; i < N; i++)
          for (j = 0; j < N; j++)
            path_lengths[i*N+j] = temp_matrix[i*N+j];

      }

    printf("\nNumber of paths of length %d:", length);
    this->print_length_matrix();

    delete [] path_lengths;
    delete [] temp_matrix;
  }

void wgraph::print_length_matrix()
  {
      printf("\n     ");
      for (int j = 0; j < n; j++)
        printf("%3d ", j);
      printf("\n   ");
      for (int j = 0; j < n; j++)
        printf("----");
      printf("\n");
      for (int i = 0; i < n; i++)
        {
          printf("%2d | ", i);
          for (int j = 0; j < n; j++)
            printf("%3d ", path_lengths[i*n + j]);
          printf("\n");
        }
  }

void wgraph::bfs_parallel(vertex root_vertex)
  {
    vertex current_vertex = root_vertex;

    for (int i = 0; i < N; i++)
      {
        vertex_levels[i] = -1;
      }

    int current_level = 0;

    vertex_levels[current_vertex] = current_level; // set level

    int level_count = 1;
    while (level_count != 0)
      {
        level_count = 0;

        #pragma omp parallel for
        for (vertex b = 0; b < N; b++) // For each vertex
          {
            if (vertex_levels[b] == current_level) // If current vertex, is of current level
              {
                // #pragma omp atomic
                level_count++;
                
                edge next_edge = first(b);

                while(next_edge != 0)
                  {
                    vertex b_neighbor = mate(b, next_edge);
                    if (vertex_levels[b_neighbor] == -1)
                      vertex_levels[b_neighbor] = current_level+1;

                    next_edge = next(b, next_edge);
                  }
              }
          }
        current_level++;
      }
  }

void wgraph::bfs_report(vertex root_vertex)
  {
    printf("\nWith vertex %d as the root:", root_vertex);

    for(int i = 0; i < N; i++)
      {
        if (vertex_levels[i] == -1)
          printf("\n  Vertex %2d is disconnected from the root vertex.", i);
        else
          printf("\n  Vertex %2d is of level %d.", i, vertex_levels[i]);
      }
    return;
  }

// Return true if every vertex in the graph is connected to the root, return false if any vertex is disconnected from the root
bool wgraph::bfs_connect_check(vertex root_vertex)
  {
    printf("\nrunning connect check");
    for(int i = 0; i < N; i++)
      {
        if (vertex_levels[i] == -1)
          return false; // Return false if a vertex in the graph is disconnected from the root vertex
      }
    return true; // Return true if all vertices in the graph are connected to the root vertex
  }


// Functional only for proper graphs
void wgraph::shortest_path(int num_vertices, vertex start, vertex end)
  {
    for (int i = 0; i < num_vertices; i++)
      tentative_distances[i] = INT_MAX; // Set starting tentative distance of all vertices to the max value
    tentative_distances[start] = 0; // Set starting vertex distance to 0

    path_member *current_vertex = (path_member *) malloc(sizeof(path_member)); // Visit queue
    path_member *queue_head = current_vertex;
    path_member *queue_tail = current_vertex;
    current_vertex->vertex_number = start;
    current_vertex->next_vertex = NULL;

    int predecessors[num_vertices]; // Used to keep track of predecessor vertices for optimal path
    for (int i = 0; i < num_vertices; i++)
      predecessors[i] = -1;

    // ------------------------ CALCULATE DISTANCES ------------------------
    int new_distance;
    while (current_vertex != NULL) // Run algorithm as long as the queue is not empty
      {
        edge current_edge = first(current_vertex->vertex_number); // Get first edge adjacent to vertex and process
        while(current_edge != 0) // If edge to be processed exists, run loop (rnext and lnext edges are initialized to 0)
          {
            vertex right = edges[current_edge].r;
            vertex left = edges[current_edge].l;

            if (right == current_vertex->vertex_number) // If right vertex of edge is current vertex
              {
                new_distance = w(current_edge) + tentative_distances[right];

                if (new_distance < tentative_distances[left]) // If new length is less than previous length, replace with new length
                  {
                    tentative_distances[left] = new_distance;
                    predecessors[left] = right;

                    //Add discovered vertex to queue and move tail
                    queue_tail->next_vertex = (path_member *) malloc(sizeof(path_member));
                    queue_tail->next_vertex->vertex_number = left;
                    queue_tail->next_vertex->next_vertex = NULL;
                    queue_tail = queue_tail->next_vertex;
                  }
              }
            else if (left == current_vertex->vertex_number) // If left vertex of edge is current vertex
              {
                new_distance = w(current_edge) + tentative_distances[left];

                if (new_distance < tentative_distances[right]) // If new length is less than previous length, replace with new length
                  {
                    tentative_distances[right] = new_distance;
                    predecessors[right] = left;

                    //Add discovered vertex to queue and move tail
                    queue_tail->next_vertex = (path_member *) malloc(sizeof(path_member));
                    queue_tail->next_vertex->vertex_number = right;
                    queue_tail->next_vertex->next_vertex = NULL;
                    queue_tail = queue_tail->next_vertex;
                  }
              } 
            current_edge = next(current_vertex->vertex_number, current_edge); // Get next edge adjacent to vertex
          }
        current_vertex = current_vertex->next_vertex; // Move to next vertex in queue
      }

    // Deallocate queue linked list
    path_member* previous = queue_head;
    path_member* current = queue_head;
    while (previous != NULL)
      {
        current = previous->next_vertex;
        free(previous);
        previous = current;
      }
    // --------------------------------------------------------------------

    // --------------- FIND SHORTEST PATH USING PREDECESSORS ---------------
    path_member *path_list = (path_member *) malloc(sizeof(path_member));
    
    shortest_path_info->path_length = tentative_distances[end];

    int current_vertex1 = end;
    path_list->vertex_number = current_vertex1;
    path_list->next_vertex = NULL;

    int predecessor;

    path_member *temp;

    while (current_vertex1 != start && current_vertex1 != -1) // predecessor array is initialized to -1
      {
        predecessor = predecessors[current_vertex1];

        temp = (path_member *) malloc(sizeof(path_member));
        temp->vertex_number = predecessor;
        temp->next_vertex = path_list;

        path_list = temp;

        current_vertex1 = predecessor;
      }

    shortest_path_info->head = path_list;
    // --------------------------------------------------------------------

    return;
  }

void wgraph::shortest_path_report()
  {
    int path_length = shortest_path_info->path_length;

    if (path_length < INT_MAX) // If vertices are connected
      {
        printf("Minimum path weight is %d.\nPath: %d", shortest_path_info->path_length, shortest_path_info->head->vertex_number);
        
        _path_member *current_vertex = shortest_path_info->head->next_vertex;
        while (current_vertex != NULL)
          {
            printf(" -> %d", current_vertex->vertex_number);
            current_vertex = current_vertex->next_vertex;
          }
        printf("\n(There may be other paths of equal length.)");
      }
    else // If vertices are disconnected
      printf("Minimum path weight is infinite (vertices are disconencted).");

    printf("\n");
    return;
  }

void wgraph::print_shortest_path_matrix()
  {
    printf("\n");
    for (int i = 0; i < n; i++)
      printf("%d  ", tentative_distances[i]);
  }

double wgraph::characteristic_path_length(int num_vertices)
  {
    unsigned long long sum = 0;
    for (vertex i = 0; i < num_vertices; i++)
      {
        this->shortest_path(num_vertices, i, i); //end vertex (third parameter) does not matter because shortest path is calculated from start to every other vertex
        for (int j = 0; j < num_vertices; j++)
          sum += tentative_distances[j];
      }
// printf("\n   %d %d", num_vertices, sum);
    return (double)sum/((double)num_vertices*(double)(num_vertices-1));
  }

double wgraph::clustering_coefficient(int num_vertices, int num_edges)
  {
    int weight_sum;
    double cluster_sum = 0;
    int num_adjacent_vertices;
    path_member *adjacent_vertices;
    path_member *new_vertex;
    path_member *previous_vertex;
    path_member *current_adjacent_vertex;
    path_member *current_adjacent_vertex2;
    edge adjacent_edge;
    bool match_found;

    for (vertex i = 0; i < num_vertices; i++)
      {
// printf("\nvertex %d:", i);
        // Create linked list of vertices adjacent to current vertex
        adjacent_vertices = NULL;
        adjacent_edge = first(i);

        num_adjacent_vertices = 0; // Count number of vertices adjacent to current vertex
        do
          {
            new_vertex = (path_member *) malloc(sizeof(path_member));
            if (adjacent_vertices == NULL)
              adjacent_vertices = new_vertex; // Set head of linked list
            else
              previous_vertex->next_vertex = new_vertex;

            if (left(adjacent_edge) == i) // If current vertex is the left vertex of the adjacent edge
              new_vertex->vertex_number = right(adjacent_edge);
            else // If current vertex is the right vertex of the adjacent edge
              new_vertex->vertex_number = left(adjacent_edge);
            // printf("  vertex %d from edge %d added\n", new_vertex->vertex_number, adjacent_edge);
            new_vertex->next_vertex = NULL;

            // Move to next adjacent edge
            if (edges[adjacent_edge].r == i)
              adjacent_edge = edges[adjacent_edge].rnext;
            else
              adjacent_edge = edges[adjacent_edge].lnext;

            previous_vertex = new_vertex;
            num_adjacent_vertices++; // Count number of vertices adjacent to current vertex
          }
        while (adjacent_edge != 0); // ------------------------
// printf("num_edges: %d\n", num_edges);
        // printf("adjacent vertices: %d\n", num_adjacent_vertices);
        // Get total weight of edges connecting vertices adjacent to current vertex 
        weight_sum = 0;
        for (edge j = 1; j <= num_edges; j++)
          {
            current_adjacent_vertex = adjacent_vertices;
            while(current_adjacent_vertex != NULL)
              {
                // printf("%d ", current_adjacent_vertex->vertex_number); // ------------------------
                match_found = false;
                current_adjacent_vertex2 = current_adjacent_vertex->next_vertex;
                while(!match_found && current_adjacent_vertex2 != NULL)
                  {
                    // printf("\n  edge %d -- %d, %d -- %d, %d", j, left(j), right(j), current_adjacent_vertex->vertex_number, current_adjacent_vertex2->vertex_number);
                    if ( (left(j) == current_adjacent_vertex->vertex_number && right(j) == current_adjacent_vertex2->vertex_number) || (right(j) == current_adjacent_vertex->vertex_number && left(j) == current_adjacent_vertex2->vertex_number) )
                      {
                        // printf("\n  adding edge %d", j);
                        weight_sum += w(j);
                        break;
                      }
                    current_adjacent_vertex2 = current_adjacent_vertex2->next_vertex;
                  }
                current_adjacent_vertex = current_adjacent_vertex->next_vertex;
              }
          }

        // printf("\n  cluster_sum: %.2lf\n  num_vertices: %.2lf\n  weight_sum: %.2lf\n  num_adjacent_vertices: %.2lf\n", (double)cluster_sum, (double)num_vertices, (double)weight_sum, (double)num_adjacent_vertices);

        if (num_adjacent_vertices > 1)
          cluster_sum += (double)weight_sum/((double)num_adjacent_vertices*((double)num_adjacent_vertices-1)/2);
        // printf("\n  cluster_sum: %.2lf\n", cluster_sum);

        // Deallocate adjacent vertex linked list
        current_adjacent_vertex = adjacent_vertices;
        while (adjacent_vertices != NULL)
          {
            current_adjacent_vertex = adjacent_vertices->next_vertex;
            free(adjacent_vertices);
            adjacent_vertices = current_adjacent_vertex;
          }
      }
    return cluster_sum/(double)num_vertices;
  }