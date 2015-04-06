Usage
=====
Usage: gpu-np-synth [option]

Options:  
    --help,-h      print this message  
    --verbose,-v   basic verbosity level  
    --debug,-d     enhanced verbosity level

Other:  
    --import,-i <graph_file>           import graph file  
    --thread,-t <number of threads>    specify number of threads  
    --format,-f <number>               specify the input format  
                 0 - DIMACS9  
                 1 - DIMACS10  
                 2 - SLNDC  
    --solution,-s <number>             specify the solution  
                 0 - Unordered + thread queue  
                 1 - dual queue  
                 2 - shared delayed buffer  
                 3 - global dedayed buffer  
                 4 - multiple dynamic parallelism per block  
                 5 - single dynamic parallelism per block  
    --device,-e <number>               select the device

Datasets
========
See DIMACS9, DIMACS10 and SLNDC graphs in ./datasets folder

Source Files
============

CPU
---
Empty

GPU
---
* np_synth.cpp - entry point, include main function and other utility functions like printing help information, parsing arguments, initializing configuration and printing configuration. The main function deals with reading data and internal format conversion.
* np_synth.h - definition of configuration and other global variables
* np_synth_wrapper.cu - implementation of preparation, clean and wrapper function  
  * preparation: GPU memory allocation, data transfer, kernel configuration initialization
  * clean: GPU memory deallocation
  * wrapper: code on CPU that perform as interface between CPU and GPU. It is the wrapper of kernel launches 
* np_synth_kernel.cu - implementation of np_synth kernels on CPU  
  The following implementations are provided:
  1. unordered + thread queue  
  2. dual queue  
  3. shared delayed buffer  
  4. global dedayed buffer  
  5. multiple dynamic parallelism per block (less efficient)
  6. single dynamic parallelism per block (efficient)
  
  Note: for large graphs, it is necessary to change the OS setting for the stack size:
  * bash command: ulimit -s unlimited
  * csh command: set stacksize unlimited

Precompiler Variables  
---------------------
- PROFILE_CPU  
- PROFILE_GPU  

Notes
==============
- N times performance measurement has not been implemented.
