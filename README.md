- src : source 
 - bc (IPDPS+Consolidation) 
 - bfs-rec (IPDPS)
 - bfs-synth  
 - graph-color (IISWC paper, debugging) 
 - join (IISWC paper, empty)
 - np-synth (IPDPS)
 - pagerank (IPDPS+Consolidation)
 - rec-synth (empty)
 - SpMV (IPDPS+Consolidation)
 - sssp (IPDPS+Consolidation)
 - tree-descendant (IPDPS)
 - tree-height (IPDPS)

To work on your applications, just reuse the sssp as template. In "sssp_wrapper.cu", function "sssp_np_consolidate_gpu" is defined, which has different consolidation schemes(warp, block, grid-level). "sssp_kernel.cu" contains all the kernel functions.
