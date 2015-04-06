Usage
=====
Usage: gpu-bc [option]

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
                 0 - thread bitmap
                 1 - thread queue
                 2 - dual queue
                 3 - shared delayed buffer  
                 4 - global dedayed buffer  
                 5 - multiple dynamic parallelism per block  
                 6 - single dynamic parallelism per block
                 7 - workload consolidation for dynamic parallelism  
    --device,-e <number>               select the device

Datasets
========
See DIMACS9, DIMACS10 and SLNDC graphs in ./datasets folder

Source Files
============

CPU
---
* bc.cpp - entry point
* bc.h - definition of configuration and other global variables
* bc_kernel.cpp - implementation of preparation, wrapper function and SSSP kernel on CPU

GPU
---
* bc.cpp - entry point, include main function and other utility functions like printing help information, parsing arguments, initializing configuration and printing configuration. The main function deals with reading data and internal format conversion.
* bc.h - definition of configuration and other global variables
* bc_wrapper.cu - implementation of preparation, clean and wrapper function  
  * preparation: GPU memory allocation, data transfer, kernel configuration initialization
  * clean: GPU memory deallocation
  * wrapper: code on CPU that perform as interface between CPU and GPU. It is the wrapper of kernel launches
* bc_fp_kernel.cu - implementation of bc forward kernels on CPU
* bc_bp_kernel.cu - implementation of bc backward kernels on CPU  
  The following implementations are provided:
  1. thread bitmap
  2. thread queue  
  3. dual queue  
  4. shared delayed buffer  
  5. global dedayed buffer  
  6. multiple dynamic parallelism per block (less efficient)
  7. single dynamic parallelism per block (efficient)
  8. workload consolidation for dynamic parallelism
  
* bc_kernel.cu - implementation of utility kernels
  
  Note: for large graphs, it is necessary to change the OS setting for the stack size:
  * bash command: ulimit -s unlimited
  * csh command: set stacksize unlimited

Precompiler Variables  
---------------------
- PROFILE_GPU
- CONSOLIDATE_LEVEL  

Notes
==============
- bc has forward phase and backward phase
