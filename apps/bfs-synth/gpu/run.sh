#!/bin/bash

./bfs 0 0 0 < ../../data/bfs/USA-road-d.COL.gr | tee -a July-02.txt
./bfs 1 0 0 < ../../data/bfs/coPapersCiteseer.graph | tee -a July-02.txt
./bfs 2 0 0 < ../../data/bfs/p2p-Gnutella31.txt | tee -a July-02.txt
./bfs 2 0 0 < ../../data/bfs/amazon0505.txt | tee -a July-02.txt
./bfs 2 0 0 < ../../data/bfs/web-Google.txt | tee -a July-02.txt
./bfs 2 0 0 < ../../data/bfs/soc-LiveJournal1.txt | tee -a July-02.txt
