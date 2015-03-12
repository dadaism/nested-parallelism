#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Used to read in parameters (strcmp())
#include "stdinc.h"
#include "stdinc.cpp"
#include "wgraph.h"
#include "wgraph.cpp"
#include "reporting.h"
#include "main_random.h"
#include "main_small_world.h"
#include "main_scale_free.h"
// #include "main_load.h"

int main(int argc, char **argv)
  {
    int program_choice;
    if (argc == 1)
      {
        printf("\nChoose program to run:"
               "\n  (1) Random graph"
               "\n  (2) Small-world"
               "\n  (3) Scale-free"
               // "\n  (4) Load graph"
               "\n  (0) Exit"
               "\n\n  Enter choice: ");
        scanf("%d", &program_choice);
        argc++; // Second argv value for main program files is the program choice
      }
    else
      program_choice = atoi(argv[1]);

    // Choose program based on value
    switch (program_choice)
      {
        case 1:
          printf("\n--Running random graph program--\n");
          main_random(argc, argv);
          break;
        case 2:
          printf("\n--Running small-world program--\n");
          main_small_world(argc, argv);
          break;
        case 3:
          printf("\n--Running scale-free program--\n");
          main_scale_free(argc, argv);
          break;
        // case 4:
        //   main_load(argc, argv);
        //   break;
        case 0:
          printf("\nExiting program. Goodbye!\n\n");
          return 0;
        default:
          printf("\nError. Program specified (%d) is not an option.\nExiting program.\n\n", program_choice);
          return 0;           
      }

    return 0;
  }