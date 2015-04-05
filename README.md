Nested Parallelism Benchmark
============================
This benchmark includes several implementations of our previous papers:

* **[IPDPS '13](http://)**
* **[GTC '15](http://)**
* **[COSMIC '15](http://)**
* **[ICPP '15](http://icpp2015.tsinghua.edu.cn)**

Top Directory Structure
-------------------
* apps: include different applications
* bin: executable binary
* common: shared .h and .c library
* datasets: graph datasets
* log: logs and output results
* run: scripts for running applications
* test: scripts for debugging testing applications
* utility: 
  * graph-generator
  * graph-analyzer
  * np-data-generator
  * np-data-analyzer
  * relational-table-generator
  * graph-code: graph generator from Thomas

Included Applications
---------------------
* bc: Betweenness Centrality
* bfs-rec: Recursive Breadth-First Search
* bfs-synth: Synthetic work Breadth-First Search
* graph-color: Graph Coloring
* join: Inner Join
* np-synth: Synthetic Nested Parallelism
* pagerank: PageRank
* rec-synth: Synthetic Recursive
* SpMV: Sparse Matrix Vector Multiplication
* sssp: Single Source Shortest Path
* tree-descendant: Tree Descendant
* tree-height: Tree Height

Create New Applications
--------------------
To create a new application, just reuse the sssp as template. In "sssp_wrapper.cu", function "sssp_np_consolidate_gpu" is defined, which has different consolidation schemes(warp, block, grid-level). "sssp_kernel.cu" contains all the kernel functions.

Notes
-----