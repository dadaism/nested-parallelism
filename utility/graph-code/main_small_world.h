// This is the main program for the small-world network program
int main_small_world(int argc, char **argv)
  {
    int num_vertices;
    int initial_degree;
    int max_num_edges;
    int max_weight;
    double probability;
    graph_type_ graph_type;
    char data_file_lattice[] = "mygraph_generated_small_world_lattice.txt";
    char dot_file_lattice[] = "mygraph_generated_small_world_lattice.dot"; // If filename is changed, corresponding system() call must also be changed
    char data_file_random[] = "mygraph_generated_small_world_random.gr";
    char dot_file_random[] = "mygraph_generated_small_world_random.dot"; // If filename is changed, corresponding system() call must also be changed
    
    // -------- SET PARAMETERS --------
    if (argc != 2 && argc != 7) // Accept only 0 or 5 arguments from command line (in addition to program command and program to run)
      {
        printf("\nError.  Incorrect number of inputs."
               "\nEnter either: %s\nor: %s %c [num_vertices] [initial_degree] [max_weight] [probability] [graph_type]\n\n", argv[0], argv[0], *argv[1]);
        exit(1);
      }
    if (argc == 2) // If user does not specify parameters
      {
        // Get num vertices, max degree, max weight
        printf("\nEnter number of vertices: ");
        scanf("%d", &num_vertices);
        printf("Enter the initial degree for each vertex (must be even): ");
        scanf("%d", &initial_degree);
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
                    "\n  proper graph: disallows self-loops and disallows parallel edges");
                need_type = true;
              }
            else
              {
                printf("Error. Provided parameter is not an option. Try again.\n");
                need_type = true;
              }
          }
      }
    else if (argc == 7) // If user does specify parameters; parameters are assumed to be valid values, invalid values may produce unexpected results
      {
        num_vertices = atoi(argv[2]);
        initial_degree = atoi(argv[3]);
        max_weight = atoi(argv[4]);
        probability = atof(argv[5]);
        char graph_type_input = *argv[6];
        (graph_type_input == 'm') ? graph_type = multigraph : graph_type = proper;
      }

    while (num_vertices <= 0 || (initial_degree % 2 != 0) || num_vertices <= initial_degree)
    {
      printf("\nThe number of vertices must be greater than 0, the initial degree must be even, and the number of vertices must be greater than the initial degree.");
      printf("\nEnter a new value for the number of vertices: ");
      scanf("%d", &num_vertices);
      printf("Enter a new value for the initial degree for each vertex: ");
      scanf("%d", &initial_degree);
    }

    // -------- CREATE GRAPH --------
    max_num_edges = initial_degree * num_vertices; //  Used to make sure the array of edge structures is large enough to accomodate input parameters
    wgraph *graph_lattice = new wgraph(num_vertices, max_num_edges);

    // Create normal ring lattice
    graph_lattice->small_world_lattice(num_vertices, initial_degree, max_weight);

    // Save data file and dot file, and create image
    // FILE *save_file = fopen(data_file_lattice, "w");
    // graph_lattice->save(save_file, num_vertices, graph_lattice->joined_edge_count, initial_degree, max_weight, false, probability, 0, graph_type);
    // fclose(save_file);
    // graph_lattice->undirected_to_dot(dot_file_lattice);
    // system("dot -Tjpeg mygraph_generated_small_world_lattice.dot > mygraph_generated_small_world_lattice.jpeg -q"); // '-q' suppresses warning messages.
    // system("neato -Tjpeg mygraph.dot > mygraph_small_world_lattice.jpeg -q"); // '-q' suppresses warning messages.

    // Get probability value for randomization
    if (argc == 2) // If probability value was not provided in command line, get it here
      {
        printf("\nEnter probability value between 0 and 1 (inclusive) for randomization: ");
        scanf("%lf", &probability);
      }
    while (probability < 0 || probability > 1)
    {
      printf("\nThe edge randomization probability must have a value between 0 and 1.");
      printf("\nEnter a new value for the probability: ");
      scanf("%lf", &probability);
    }

    // Duplicate graph object to allow for computaton on both lattice and randomized graph
    wgraph *graph_randomized = new wgraph(graph_lattice);

    // Randomize ring lattice given probability value
    srand(time(NULL));
    graph_randomized->small_world_randomize(num_vertices, max_num_edges, probability, false, graph_type);

    // Save data file
    FILE *save_file = fopen(data_file_random, "w");
    graph_randomized->save_gr(save_file, num_vertices, graph_randomized->joined_edge_count, initial_degree, max_weight, false, probability, graph_randomized->rewired_edge_count, graph_type);
    fclose(save_file);

    // Save dot file and create image
    // graph_randomized->undirected_to_dot(dot_file_random);
    // system("dot -Tjpeg mygraph_generated_small_world_random.dot > mygraph_generated_small_world_random.jpeg -q"); // '-q' suppresses warning messages.
    // system("neato -Tjpeg mygraph.dot > mygraph_small_world_random.jpeg -q"); // '-q' suppresses warning messages.

    graph_report_small_world(graph_lattice, graph_randomized, num_vertices, max_num_edges, initial_degree, max_weight, false, probability, graph_randomized->rewired_edge_count, graph_type, data_file_lattice, data_file_random, dot_file_lattice, dot_file_random, true);

    delete graph_lattice;
    delete graph_randomized;
    return 0;
  }