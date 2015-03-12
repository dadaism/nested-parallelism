// This is the main program for loading a graph from a data file 
int main_load(int, char**);
void get_edges(wgraph*, int, FILE*);

int main_load(int argc, char **argv)
  {
    char *file_name;
    if (argc == 2)
      {
        file_name = (char *) malloc(sizeof(char) * 100);
        printf("\nEnter name of input file: ");
        scanf("%s", file_name);
      }
    else
      file_name = argv[2];

    FILE *data_file = fopen(file_name, "r");

    if (data_file) // Run only if file is found
      {
        char graph_characteristic, graph_type;
        int num_vertices, num_edges;

        fscanf(data_file, " %c %c %d %d", &graph_characteristic, &graph_type, &num_vertices, &num_edges);
        wgraph *graph = new wgraph(num_vertices, num_edges);

        if (graph_characteristic == 'r') // If graph is random
          {
            char dot_file[] = "mygraph_loaded_random.dot"; // If filename is changed, corresponding system() call must also be changed
            int max_degree, max_weight;
            fscanf(data_file, " %d %d", &max_degree, &max_weight);

            char directed_graph_char, force_connected_char, allow_leaves_char;
            fscanf(data_file, " %c %c %c", &directed_graph_char, &force_connected_char, &allow_leaves_char);
            bool directed_graph = directed_graph_char == 't' ? true : false;
            bool force_connected = force_connected_char == 't' ? true : false;
            bool allow_leaves = allow_leaves_char == 't' ? true : false;
            
            get_edges(graph, num_edges, data_file);

            graph->undirected_to_dot(dot_file);
            system("dot -Tjpeg mygraph_loaded_random.dot > mygraph_loaded_random.jpeg -q"); // '-q' suppresses warning messages.

            graph_report_random(graph, num_vertices, num_edges, max_degree, max_weight, directed_graph, force_connected, allow_leaves, graph_type, NULL, dot_file, false);
          }
        else if (graph_characteristic == 'w') // If graph is a small-world network
          {
            char dot_file_lattice[] = "mygraph_loaded_small_world_lattice.dot"; // If filename is changed, corresponding system() call must also be changed
            char dot_file_random[] = "mygraph_loaded_small_world_random.dot"; // If filename is changed, corresponding system() call must also be changed
            int initial_degree, max_weight, rewired_count;
            double probability;
            fscanf(data_file, " %d %d %lf %d", &initial_degree, &max_weight, &probability, &rewired_count);
            char directed_graph_char;
            fscanf(data_file, " %c", &directed_graph_char);
            bool directed_graph = directed_graph_char == 't' ? true : false;
            
            get_edges(graph, num_edges, data_file);
            fclose(data_file);

            graph->undirected_to_dot(dot_file_lattice);
            system("dot -Tjpeg mygraph_loaded_small_world_lattice.dot > mygraph_loaded_small_world_lattice.jpeg -q"); // '-q' suppresses warning messages

            if (argc < 4)
              {
                printf("\nEnter name of second input file: ");
                scanf("%s", file_name);
              }
            else
              file_name = argv[3];

            data_file = fopen(file_name, "r");

            fscanf(data_file, " %c %c %d %d", &graph_characteristic, &graph_type, &num_vertices, &num_edges);
            wgraph *graph_randomized = new wgraph(num_vertices, num_edges);
            fscanf(data_file, " %d %d %lf %d", &initial_degree, &max_weight, &probability, &rewired_count);
            fscanf(data_file, " %c", &directed_graph_char);

            get_edges(graph_randomized, num_edges, data_file);
            fclose(data_file);
            graph_randomized->undirected_to_dot(dot_file_random);
            system("dot -Tjpeg mygraph_loaded_small_world_random.dot > mygraph_loaded_small_world_random.jpeg -q"); // '-q' suppresses warning messages

            graph_report_small_world(graph, graph_randomized, num_vertices, num_edges, initial_degree, max_weight, directed_graph, probability, rewired_count, graph_type, NULL, NULL, dot_file_lattice, dot_file_random, false);
            
            delete graph_randomized;
          }
        else if (graph_characteristic == 's') // If graph is scale-free
          {
            char dot_file[] = "mygraph_loaded_scale_free.dot"; // If filename is changed, corresponding system() call must also be changed
            int max_weight;
            double gamma;
            char directed_graph_char;
            fscanf(data_file, " %d %lf %c", &max_weight, &gamma, &directed_graph_char);

            bool directed_graph = directed_graph_char == 't' ? true : false;
            
            get_edges(graph, num_edges, data_file);

            graph->undirected_to_dot(dot_file);
            system("dot -Tjpeg mygraph_loaded_scale_free.dot > mygraph_loaded_scale_free.jpeg -q"); // '-q' suppresses warning messages.

            graph_report_scale_free(graph, num_vertices, max_weight, gamma, directed_graph, NULL, dot_file, false, NULL);
          }

        delete graph;

        return 0;
      }
    else
      {
        printf("\nNo file '%s' found.\nExiting program.\n\n", file_name);
        exit(0);
      }
  }

void get_edges(wgraph *graph, int num_edges, FILE *data_file)
  {
    // printf("\n\n");
    int left, right, weight;
    for (int i = 0; i < num_edges; i++)
      {
        if (fscanf(data_file, " %d %d %d", &left, &right, &weight) == EOF)
          {
            printf("\nError reading input data file.\nExiting program.\n\n");
            delete graph;
            fclose(data_file);
            exit(0);
          }
// printf("%d %d %d\n", left, right, weight);
        graph->join(left, right, weight);
        graph->bld_adj_mat(left, right);
      }
  }