// This is the main program for the random graph generator
int main_random(int argc, char **argv)
  {
    int num_vertices;
    int min_degree;
    int max_degree;
    int max_num_edges; //  Used to make sure the array of edge structures is large enough to accomodate input parameters
    int max_weight;
    bool directed_graph; // Set to true for directed graph, false for undirected graph.
    bool allow_leaves;
    graph_type_ graph_type; // Accepts 'multigraph' and 'proper'
    char data_file[] = "mygraph_generated_random.gr";
    char dot_file[] = "mygraph_generated_random.dot"; // If filename is changed, corresponding system() call must also be changed
    
    // -------- SET PARAMETERS --------
    if (argc != 2 && argc != 8) // Accept only 0 or 6 arguments from command line (in addition to program command and program to run)
      {
        printf("\nError. Incorrect number of inputs.\nEnter either: %s\nor: %s %c [num_vertices] [min_degree] [max_degree] [max_weight] [graph_type] [directed]\n\n", argv[0], argv[0], *argv[1]);
        exit(1);
      }
    if (argc == 2) // If user does not specify parameters
      {
        // Get num vertices, max degree, max weight
        printf("\nEnter number of vertices: ");
        scanf("%d", &num_vertices);
        printf("Enter the minimum out-degree for each vertex: ");
        scanf("%d", &min_degree);
        printf("Enter the maximum out-degree for each vertex: ");
        scanf("%d", &max_degree);
        printf("Enter the maximum weight for each edge: ");
        scanf("%d", &max_weight);

        // Get graph type (multigraph, undirected, or proper)
        bool need_type = true;
        while (need_type)
          {
            printf("What type of graph? Enter m for multigraph, p for proper graph, or ? to see definitions of graph types: ");
            char graph_type_input;
            scanf(" %c", &graph_type_input); // Accept 'm' for multigraph, or 'p' for proper graph
            need_type = false;
            if (graph_type_input == 'm')
              graph_type = multigraph;
            else if (graph_type_input == 'p')
              graph_type = proper;
            else if (graph_type_input == '?')
              {
                printf("For undirected graphs, the types of graphs are:"
                    "\n  multigraph: allows both self-loops (an edge from a vertex to the same vertex)"
                    "\n     and allows parallel edges (multiple edges between the same two vertices)"
                    "\n  proper graph: disallows self-loops and disallows parallel edges"
                    "\nFor directed graphs, the types of graphs are:"
                    "\n  multigraph: allows both self-loops and parallel edges"
                    "\n  proper graph: allows self-loops and disallows parallel edges (multiple edges"
                    "\n     having the same orientation between the same two vertices)");
                need_type = true;
              }
            else
              {
                printf("Error. Provided parameter is not an option. Try again.\n");
                need_type = true;
              }
          }

        // Get directed/undirected attribute
        char directed_input;
        printf("Directed or undirected graph? Enter d or u: ");
        scanf(" %c", &directed_input);
        bool need_directed = true;
        while(need_directed)
          {
            need_directed = false;
            if (directed_input == 'd')
              directed_graph = true;
            else if (directed_input == 'u')
              directed_graph = false;
            else
              {
                printf("Error. Provided parameter is not an option. Try again.\n");
                need_directed = true;
              }
          }
      }
    else if (argc == 8) // If user does specify parameters; parameters are assumed to be valid values, invalid values may produce unexpected results
      {
        int arg_num = 2;
        num_vertices = atoi(argv[arg_num++]);
        min_degree = atoi(argv[arg_num++]);
        max_degree = atoi(argv[arg_num++]);
        max_weight = atoi(argv[arg_num++]);
        char graph_type_input = *argv[arg_num++];
        (graph_type_input == 'm') ? graph_type = multigraph : graph_type = proper;
        char directed_graph_input = *argv[arg_num++];
        (directed_graph_input == 'u') ? directed_graph = false : directed_graph = true;
      }

      // Check parameters
      if (!directed_graph && graph_type == proper) // Check parameters for undirected proper graphs
        {
          while ((min_degree < 1) || (max_degree < 2) || (num_vertices <= max_degree) || (max_degree < min_degree))
            {
              printf("\nFor a connected undirected proper graph, the min out-degree must be greater than 0, the max out-degree must be greater"
                     "\nthan 1, the max out-degree must be greater than or equal to the min out-degree, and the max out-degree must be less"
                     "\nthan the number of vertices.  Please enter parameters that fit these requirements."
                     "\nEnter a new minimum out-degree that is 1 or greater: ");
              scanf("%d", &min_degree);
              printf("\nEnter a maximuim out-degree that is greater than ");
              if (min_degree >= 1) printf("or equal to ");
              printf("%d, and less than %d: ", min_degree, num_vertices);
              scanf("%d", &max_degree);
            }
        }
      else if (!directed_graph && graph_type == multigraph) // Check parameters for undirected multigraphs
        {
          while ((max_degree < min_degree) || (min_degree < 1) || (max_degree < 2))
            {
              printf("\nFor a connected undirected multigraph, the min out-degree must be greater than 0, the max out-degree must be greater"
                     "\nthan 1, and the max out-degree must be greater than or equal to the min out-degree.  Please enter parameters that"
                     "\nfit these requirements."
                     "\nEnter a new minimum out-degree that is 1 or greater: ");
              scanf("%d", &min_degree);
              printf("\nEnter a maximuim out-degree that is greater than ");
              if (min_degree >= 1) printf("or equal to ");
              printf("%d: ", min_degree);
              scanf("%d", &max_degree);
            }
        }
      else if (directed_graph && graph_type == proper) // Check parameters for directed proper graphs
        {
          while ((max_degree < min_degree) || (max_degree < 1) || (num_vertices <= max_degree))
            {
              printf("\nFor a connected directed proper graph, the max out-degree must be greater than 0, the max out-degree must be"
                     "\ngreater than or equal to the min out-degree, and the max out-degree must be less than the number of vertices."
                     "\nPlease enter parameters that fit these requirements."
                     "\nEnter a new minimum out-degree: ");
              scanf("%d", &min_degree);
              printf("\nEnter a maximuim out-degree that is greater than ");
              if (min_degree > 0) printf("or equal to ");
              printf("%d, and less than %d: ", min_degree, num_vertices);
              scanf("%d", &max_degree);
            }
        }
      else if (!directed_graph && graph_type == multigraph) // Check parameters for directed multigraphs
        {
          while ((max_degree < min_degree) || (min_degree < 0) || (max_degree < 2))
            {
              printf("\nFor a connected directed multigraph, the min out-degree must be equal to or greater than 0, the max out-degree"
                     "\nmust be greater than 0, and the max out-degree must be greater than or equal to the min out-degree."
                     "\nPlease enter parameters that fit these requirements."
                     "\nEnter a new minimum out-degree: ");
              scanf("%d", &min_degree);
              printf("\nEnter a maximuim out-degree that is greater than ");
              if (min_degree > 0) printf("or equal to ");
              printf("%d: ", min_degree);
              scanf("%d", &max_degree);
            }
        }

    // Generate graph and output files
    max_num_edges = max_degree * num_vertices; //  Used to make sure the array of edge structures is large enough to accomodate input parameters

    wgraph *graph = new wgraph(num_vertices, max_num_edges);

    if (!directed_graph)
      graph->random_graph_undirected(num_vertices, min_degree, max_degree, max_weight, graph_type);
    else
      graph->random_graph_directed(num_vertices, min_degree, max_degree, max_weight, graph_type);

    // Save data file
    FILE *save_file = fopen(data_file, "w");
    graph->save_gr(save_file, num_vertices, graph->joined_edge_count, min_degree, max_degree, max_weight, directed_graph, graph_type);
    fclose(save_file);

    // Save dot file and create image
    // directed_graph ? graph->directed_to_dot(dot_file) : graph->undirected_to_dot(dot_file);
    // system("dot -Tjpeg mygraph_generated_random.dot > mygraph_generated_random.jpeg -q"); // '-q' suppresses warning messages.
    // system("neato -Tjpeg mygraph.dot > mygraph_random_neato.jpeg -q");

    graph_report_random(graph, num_vertices, graph->joined_edge_count, min_degree, max_degree, max_weight, directed_graph, graph_type, data_file, dot_file, true);

    delete graph;
  	
    return 0;
  }