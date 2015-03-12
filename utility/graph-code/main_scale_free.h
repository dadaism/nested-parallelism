// This is the main program for the scale-free network program
bool check_parameters(int, double);

int main_scale_free(int argc, char **argv)
  {
    int num_vertices;
    int max_weight;
    int max_num_edges;
    double gamma;
    char data_file[] = "mygraph_generated_scale_free.gr";
    char dot_file[] = "mygraph_generated_scale_free.dot";
    
    // -------- SET PARAMETERS --------
    if (argc != 2 && argc != 5) // Accept only 0 or 3 arguments from command line (in addition to program command and program to run)
      {
        printf("\nError.  Incorrect number of inputs."
               "\nEnter either: %s\nor: %s %c [num_vertices] [max_weight] [gamma]\n\n", argv[0], argv[0], *argv[1]);
        exit(1);
      }
    if (argc == 2) // If user does not specify parameters
      {
        // Get num vertices, max degree, max weight
        printf("\nEnter number of vertices: ");
        scanf("%d", &num_vertices);
        printf("Enter the maximum weight for each edge: ");
        scanf("%d", &max_weight);
        printf("Enter the gamma value: ");
        scanf("%lf", &gamma);
      }
    else if (argc == 5) // If user does specify parameters; parameters are assumed to be valid values, invalid values may produce unexpected results
      {
        num_vertices = atoi(argv[2]);
        max_weight = atoi(argv[3]);
        gamma = atof(argv[4]);
      }

    // -------- CREATE GRAPH --------
    max_num_edges = num_vertices*num_vertices/2; //  Used to make sure the array of edge structures is large enough to accomodate input parameters

    wgraph *graph = new wgraph(num_vertices, max_num_edges);

    // Create scale-free network
    srand(time(NULL));

    scale_free_info *sfi = graph->scale_free(num_vertices, max_weight, gamma);

    // Save data file
    FILE *save_file = fopen(data_file, "w");
    graph->save_gr(save_file, num_vertices, graph->joined_edge_count, max_weight, gamma, false);
    fclose(save_file);

    // Save dot file and create image
    // graph->undirected_to_dot(dot_file);
    // system("dot -Tjpeg mygraph_generated_scale_free.dot > mygraph_generated_scale_free.jpeg -q"); // '-q' suppresses warning messages.
    // system("neato -Tjpeg mygraph.dot > mygraph_scale_free.jpeg -q"); // '-q' suppresses warning messages.

    graph_report_scale_free(graph, num_vertices, max_weight, gamma, false, data_file, dot_file, true, sfi);

    delete graph;
    return 0;
  }
