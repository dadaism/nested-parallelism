// This is the library file for the graph reporting functions
#ifndef REPORTING_H
#define REPORTING_H

void graph_report_random(wgraph *, int, int, int, int, int, bool, graph_type_, char *, char *, bool);
// void graph_report_small_world(wgraph *, wgraph *, int, int, int, int, bool, double, int, char, char *, char *, char *, char *, bool);
void graph_report_scale_free(wgraph *, int, int, double, bool, char *, char *, bool, scale_free_info *);

// int report_menu();
// void path_length_report(wgraph *);
// void vertex_level_report(wgraph *, int);
// void shortest_path_report(wgraph *, int);
// void characteristic_path_length_report(wgraph *, int, bool, char);
// void characteristic_path_length_report(wgraph *, wgraph *, int, bool, char);
// void clustering_coefficient_report(wgraph *, int, int, char, bool);
// void clustering_coefficient_report(wgraph *, wgraph *, int, int, char, bool);

void graph_report_random(wgraph * graph, int num_vertices, int num_edges, int min_degree, int max_degree, int max_weight, bool directed_graph, graph_type_ graph_type, char * data_file, char * dot_file, bool generated) // For generated, pass true if generated graph and false if loaded graph
  {
    generated ? printf("\nCreating a connected random ") : printf("\nLoading a connected random ");
    directed_graph ? printf("directed ") : printf("undirected ");
    (graph_type == multigraph) ? printf("multigraph ") : printf("proper graph ");
    printf("with %d vertices,\n   each with a minimum out-degree of %d and a maximum out-degree of %d.", num_vertices, min_degree, max_degree);
    printf("\nEdges have a maximum weight of %d.", max_weight);
    printf("\nThere are %d edges.", num_edges);

    if (generated)
      printf("\nSaving graph data to %s.", data_file);
    // printf("\nSaving dot file as %s.", dot_file);
    // printf("\nSaving image file as %s.\n", generated ? "mygraph_generated_random.jpeg" : "mygraph_loaded_random.jpeg");

    // bool exit_program = false;
    // while (!exit_program)
    //   {
    //     int user_input = report_menu();
    //     switch (user_input)
    //       {
    //         case 1:
    //           path_length_report(graph);
    //           break;
    //         case 2:
    //           vertex_level_report(graph, num_vertices);
    //           break;
    //         case 3:
    //           shortest_path_report(graph, num_vertices);
    //           break;
    //         case 4:
    //           characteristic_path_length_report(graph, num_vertices, force_connected, graph_type);
    //           break;
    //         case 5:
    //           clustering_coefficient_report(graph, num_vertices, num_edges, graph_type, force_connected);
    //           break;
    //         case 6:
    //           exit_program = true;
    //           break;
    //       }
    //   }

    printf("\nExiting program. Goodbye!\n\n");
    return;
  }

void graph_report_small_world(wgraph *graph_lattice, wgraph *graph_randomized, int num_vertices, int num_edges, int initial_degree, int max_weight, bool directed_graph, double probability, int rewired_count, char graph_type, char * data_file_lattice, char * data_file_random, char * dot_file_lattice, char * dot_file_random, bool generated)
  {
    generated ? printf("\nCreating a small-world ") : printf("\nLoading a small-world ");
    directed_graph ? printf("directed ") : printf("undirected ");
    if (graph_type == multigraph)
      printf("multigraph ");
    else
      printf("proper graph ");
    printf("with %d vertices, each with an initial degree of %d.", num_vertices, initial_degree);
    printf("\nEdges have a maximum weight of %d.", max_weight);
    printf("\nThere are %d edges.\n\n", graph_lattice->joined_edge_count);

    if (generated) printf("Rewired ");
    printf("%d out of %d edges (%.4f%%)", rewired_count, graph_lattice->joined_edge_count, ((float)rewired_count/(float)graph_lattice->joined_edge_count)*100);
    if (!generated) printf(" were rewired");
    printf(".\n  Some edges may have been rewired to their original vertex (counted as rewired)\n  and it may have been impossible to rewire some edges if the initial degree\n  was close to the number of vertices (not counted as rewired).\n");

    if (generated)
      {
        // printf("\nSaving lattice graph data to %s.", data_file_lattice);
        printf("\nSaving randomized graph data to %s", data_file_random);
      }
      
//     printf("\nSaving lattice dot file as %s.\nSaving randomized dot file as %s", dot_file_lattice, dot_file_random);
//     printf("\nSaving lattice image file as %s.", generated ? "mygraph_generated_small_world_lattice.jpeg" : "mygraph_loaded_small_world_lattice.jpeg");
//     printf("\nSaving randomized image file as %s.\n", generated ? "mygraph_generated_small_world_random.jpeg" : "mygraph_loaded_small_world_random.jpeg");

//     bool exit_program = false;
//     while (!exit_program)
//       {
//         int user_input = report_menu();
//         switch (user_input)
//           {
//             case 1:
//               path_length_report(graph_randomized);
//               break;
//             case 2:
//               vertex_level_report(graph_randomized, num_vertices);
//               break;
//             case 3:
//               shortest_path_report(graph_randomized, num_vertices);
//               break;
//             case 4:
//               characteristic_path_length_report(graph_lattice, graph_randomized, num_vertices, true, graph_type);
//               break;
//             case 5:
//               clustering_coefficient_report(graph_lattice, graph_randomized, num_vertices, num_edges, graph_type, true);
//               break;
//             case 6:
//               exit_program = true;
//               break;
//           }
//       }

    printf("\nExiting program. Goodbye!\n\n");
    return;
  }

void graph_report_scale_free(wgraph *graph, int num_vertices, int max_weight, double gamma, bool directed_graph, char * data_file, char * dot_file, bool generated, scale_free_info *sfi)
  {
    generated ? printf("\nCreating a connected scale-free ") : printf("\nLoading a connected scale-free ");
    directed_graph ? printf("directed ") : printf("undirected ");
    printf("proper graph with %d vertices.", num_vertices);
    printf("\nEdges have a maximum weight of %d.", max_weight);
    printf("\nThere are %d edges.\n", graph->joined_edge_count);

    if (generated)
      printf("\nSaving graph data to %s.", data_file);
    // printf("\nSaving dot file as %s.", dot_file);
    // printf("\nSaving image file as %s.\n", generated ? "mygraph_generated_scale_free.jpeg" : "mygraph_loaded_scale_free.jpeg");

    if (sfi)
      {
        printf("\nDegree  Expected Number  Actual Number  Actual/Expected\n          of Vertices     of Vertices\n");
        scale_free_info *sfi_previous = sfi;
        while (sfi)
          {
            printf("%4d %12d %15d %16.3lf\n", sfi->degree, (int)sfi->expected_num_vertices_with_degree, sfi->num_vertices_with_degree, sfi->percentage);
            sfi = sfi->next;
            delete sfi_previous;
            sfi_previous = sfi;
          }
      }


//     bool exit_program = false;
//     while (!exit_program)
//       {
//         int user_input = report_menu();
//         switch (user_input)
//           {
//             case 1:
//               path_length_report(graph);
//               break;
//             case 2:
//               vertex_level_report(graph, num_vertices);
//               break;
//             case 3:
//               shortest_path_report(graph, num_vertices);
//               break;
//             case 4:
//               characteristic_path_length_report(graph, num_vertices, true, 'p');
//               break;
//             case 5:
//               clustering_coefficient_report(graph, num_vertices, graph->joined_edge_count, 'p', true);
//               break;
//             case 6:
//               exit_program = true;
//               break;
//           }
//       }

    printf("\nExiting program. Goodbye!\n\n");
    return;
  }

// int report_menu()
//   {
//     int user_input;
//     printf("\nOptions:"
//           "\n  (1) Display number of paths between vertices of specified\n       length (where each edge contributes 1 to the length)"
//           "\n  (2) Display vertex levels relative to specified root vertex"
//           "\n  (3) Find the path of minimum weight between two vertices"
//           "\n  (4) Calculate the characteristic path length"
//           "\n  (5) Calculate the clustering coefficient"
//           "\n  (6) Exit"
//           "\n  Enter choice: ");
//     scanf(" %d", &user_input);
//     if (user_input < 1 || user_input > 6)
//       {
//         printf("\nError. Provided input is not an option. Try again.");
//         user_input = report_menu();
//       }
//     return user_input;
//   }

// void path_length_report(wgraph * graph)
//   {
//     int path_length;
//     printf("\nEnter path length: ");
//     scanf(" %d", &path_length);

//     graph->path_length(path_length);

//     return;
//   }

// void vertex_level_report(wgraph * graph, int num_vertices)
//   {
//     int root_vertex;
//     bool continue_loop = true;
//     while(continue_loop)
//       {
//         printf("\nEnter root vertex: ");
//         scanf("%d", &root_vertex);

//         if (root_vertex > num_vertices - 1 || root_vertex < -1) // Check to see if specified vertex is in graph
//           printf("\nError.  Specified root vertex does not exist in graph.\nTry again.\n");
//         else
//           {
//             graph->bfs_parallel(root_vertex);
//             graph->bfs_report(root_vertex);
//             printf("\n");
//             continue_loop = false;
//           }
//       }
//   }

// void shortest_path_report(wgraph *graph, int num_vertices)
//   {
//     int start, end;

//     printf("\nEnter the starting vertex: ");
//     scanf("%d", &start);
//     while (start >= num_vertices || start < 0)
//       {
//         printf("Error. Specified start vertex does not exist in graph. Try again.\nEnter the starting vertex: ");
//         scanf("%d", &start);
//       }
//     printf("Enter the ending vertex: ");
//     scanf("%d", &end);
//     while (end >= num_vertices || end < 0 || end == start)
//       {
//         while (end >= num_vertices || end < 0)
//           {
//             printf("Error. Specified end vertex does not exist in graph. Try again.\nEnter the ending vertex: ");
//             scanf("%d", &end);
//           }
//         while (end == start)
//           {
//             printf("Error. End vertex must be different from start vertex. Try again.\nEnter the ending vertex: ");
//             scanf("%d", &end);
//           }
//       }
//     graph->shortest_path(num_vertices, start, end);
//     graph->shortest_path_report();
//     return; 
//   }

// void characteristic_path_length_report(wgraph *graph, int num_vertices, bool force_connected, char graph_type)
//   {
//     if (graph_type == 'p') // Necessary because algorithm is buggyu - only works for proper graphs
//       {
//         if (force_connected)
//           printf("\nThe characteristic path length is %f\n", graph->characteristic_path_length(num_vertices));
//         else
//           printf("\nThe characteristic path length cannot be calculated for graphs which may be\ndisconnected, as the distance between disconnected vertices is infinite.\nForce connectivity to calculate characteristic path lengths.\n");
//       }
//     else
//       printf("\nThe characteristic path length can only be calculated for a proper graph.\n");
//   }

// void characteristic_path_length_report(wgraph *graph_lattice, wgraph *graph_randomized, int num_vertices, bool force_connected, char graph_type)
//   {
//     if (graph_type == 'p') // Necessary because algorithm is buggyu - only works for proper graphs
//       {
//         if (force_connected)
//           {
//             // double lattice_path = graph_lattice->characteristic_path_length(num_vertices);
//             // double random_path = graph_randomized->characteristic_path_length(num_vertices);
//             // printf("\npath ratio: %.3f", random_path/lattice_path);
//             printf("\nLattice: The characteristic path length is %f", graph_lattice->characteristic_path_length(num_vertices));
//             printf("\nRandomized: The characteristic path length is %f\n", graph_randomized->characteristic_path_length(num_vertices));
//           }
//         else
//           printf("\nThe characteristic path length cannot be calculated for graphs which may be\ndisconnected, as the distance between disconnected vertices is infinite.\nForce connectivity to calculate characteristic path lengths.\n");
//       }
//     else
//       printf("\nThe characteristic path length can only be calculated for a proper graph.\n");
//   }

// void clustering_coefficient_report(wgraph *graph, int num_vertices, int num_edges, char graph_type, bool force_connected)
//   {
//     if (graph_type == 'p' && force_connected)
//       printf("\nThe clustering coeffient is %f\n", graph->clustering_coefficient(num_vertices, num_edges));
//     else if (graph_type != 'p' && !force_connected)
//       printf("\nThe clustering coeffient can only be calculated for connected proper graphs.\n");
//     else if (!force_connected)
//       printf("\nThe clustering coeffient can only be calculated for connected graphs.\nForce connectivity to calculate clustering coefficients.\n");
//     else
//       printf("\nThe clustering coeffient can only be calculated for proper graphs.\n");
//   }

// void clustering_coefficient_report(wgraph *graph_lattice, wgraph *graph_randomized, int num_vertices, int num_edges, char graph_type, bool force_connected)
//   {
//     if (graph_type == 'p' && force_connected)
//       {
//         // double lattice_cluster = graph_lattice->clustering_coefficient(num_vertices, num_edges);
//         // double random_cluster = graph_randomized->clustering_coefficient(num_vertices, num_edges);
//         // printf("\ncluster ratio: %.3f", random_cluster/lattice_cluster);
//         printf("\nLattice: The clustering coeffient is %f", graph_lattice->clustering_coefficient(num_vertices, num_edges));
//         printf("\nRandomized: The clustering coeffient is %f\n", graph_randomized->clustering_coefficient(num_vertices, num_edges));
//       }
//     else if (graph_type != 'p' && !force_connected)
//       printf("\nThe clustering coeffient can only be calculated for connected proper graphs.\n");
//     else if (!force_connected)
//       printf("\nThe clustering coeffient can only be calculated for connected graphs.\nForce connectivity to calculate clustering coefficients.\n");
//     else
//       printf("\nThe clustering coeffient can only be calculated for proper graphs.\n");
//   }

#endif /* REPORTING_H */