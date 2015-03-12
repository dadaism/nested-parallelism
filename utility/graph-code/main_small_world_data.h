// This is the main program for the small-world data collection
int main_small_world_data(int argc, char **argv)
  {
    int num_vertices;
    int initial_degree;
    int max_num_edges;
    int max_weight;
    double probability;
    char graph_type;
    char *data_file_name;
    
    // -------- SET PARAMETERS --------
    if (argc != 8) // Accept only 6 arguments from command line (in addition to program command and program to run)
      {
        printf("\nError.  Incorrect number of inputs."
               "\nEnter either: %s\nor: %s [num_vertices] [initial_degree] [max_weight] [probability] [graph_type] [data_file_name]\n\n", argv[0], argv[0]);
        exit(1);
      }
    else if (argc == 8) // If user does specify parameters; parameters are assumed to be valid values, invalid values may produce unexpected results
      {
        num_vertices = atoi(argv[2]);
        initial_degree = atoi(argv[3]);
        max_weight = atoi(argv[4]);
        probability = atof(argv[5]);
        graph_type = *argv[6];
        data_file_name = argv[7];
      }

    while (num_vertices <= 0 || (initial_degree % 2 != 0) || num_vertices <= initial_degree)
    {
      printf("\nThe number of vertices must be greater than 0, the initial degree must be even, and the number of vertices must be greater than the initial degree.");
      printf("\nEnter a new value for the number of vertices: ");
      scanf("%d", &num_vertices);
      printf("Enter a new value for the initial degree for each vertex: ");
      scanf("%d", &initial_degree);
    }
    // Get probability value for randomization
    while (probability < 0 || probability > 1)
    {
      printf("\nThe edge randomization probability must have a value between 0 and 1.");
      printf("\nEnter a new value for the probability: ");
      scanf("%lf", &probability);
    }

    double path_ratio_sum = 0; double temp_path_ratio = 0;
    double cluster_ratio_sum = 0; double temp_cluster_ratio = 0;
    int num_trials = 500;
    srand(time(NULL));
    for (int i = 1; i <= num_trials; i++)
      {
        // -------- CREATE GRAPH --------
        max_num_edges = initial_degree * num_vertices; //  Used to make sure the array of edge structures is large enough to accomodate input parameters

        wgraph *graph_lattice = new wgraph(num_vertices, max_num_edges);

        // Create normal ring lattice
        graph_lattice->small_world_lattice(num_vertices, initial_degree, max_weight);

        // Duplicate graph object to allow for computaton on both lattice and randomized graph
        wgraph *graph_randomized = new wgraph(graph_lattice);

        // Randomize ring lattice given probability value
        graph_randomized->small_world_randomize2(num_vertices, max_num_edges, probability, graph_type);

        int rewired_count = graph_randomized->rewired_edge_count;
        int joined_count = graph_randomized->joined_edge_count;
        
        printf("\n--trial %3d --  %d %d %d %lf", i, num_vertices, initial_degree, max_weight, probability);
        printf("\nRewired %d out of %d edges (%.4f%%).", rewired_count, joined_count, ((float)rewired_count/(float)joined_count)*100);

        double lattice_path = graph_lattice->characteristic_path_length(num_vertices);
        double random_path = graph_randomized->characteristic_path_length(num_vertices);
        temp_path_ratio = (double)random_path/(double)lattice_path;
        printf("\n  path: %.3f", temp_path_ratio);
        path_ratio_sum += temp_path_ratio;

        double lattice_cluster = graph_lattice->clustering_coefficient(num_vertices, max_num_edges);
        double random_cluster = graph_randomized->clustering_coefficient(num_vertices, max_num_edges);
        temp_cluster_ratio = (double)random_cluster/(double)lattice_cluster;
        printf("\n  cluster: %.3f", temp_cluster_ratio);
        cluster_ratio_sum += temp_cluster_ratio;
     
        printf("\n");
        delete graph_lattice;
        delete graph_randomized;
      }

    printf("\naverage cluster: %.3f\naverage path: %.3f\n\n", cluster_ratio_sum/num_trials, path_ratio_sum/num_trials);

    FILE * data_file = fopen(data_file_name, "a");
    fprintf(data_file, "%d %d %d %lf\naverage cluster: %.3f\naverage path: %.3f\n\n", num_vertices, initial_degree, max_weight, probability, cluster_ratio_sum/num_trials, path_ratio_sum/num_trials);
    fclose(data_file);

    return 0;
  }
