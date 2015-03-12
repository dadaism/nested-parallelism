#!/bin/bash

method=5
device=0

#../bin/gpu_sssp 0 ${method} ${device} < ../data/USA-road-d.COL.gr
#../bin/gpu_sssp 1 ${method} ${device} < ../data/coPapersCiteseer.graph
#../bin/gpu_sssp 2 ${method} ${device} < ../data/p2p-Gnutella31.txt 
#../bin/gpu_sssp 2 ${method} ${device} < ../data/amazon0505.txt 
#../bin/gpu_sssp 2 ${method} ${device} < ../data/web-Google.txt 
#../bin/gpu_sssp 2 ${method} ${device} < ../data/soc-LiveJournal1.txt


method=8

#../bin/gpu_sssp 1 ${method} ${device} < ../data/coPapersCiteseer.graph
../bin/gpu_sssp 2 ${method} ${device} < ../data/soc-LiveJournal1.txt

method=9

#../bin/gpu_sssp 1 ${method} ${device} < ../data/coPapersCiteseer.graph
../bin/gpu_sssp 2 ${method} ${device} < ../data/soc-LiveJournal1.txt

method=10

#../bin/gpu_sssp 1 ${method} ${device} < ../data/coPapersCiteseer.graph
#../bin/gpu_sssp 2 ${method} ${device} < ../data/soc-LiveJournal1.txt

method=11

#../bin/gpu_sssp 1 ${method} ${device} < ../data/coPapersCiteseer.graph
../bin/gpu_sssp 2 ${method} ${device} < ../data/soc-LiveJournal1.txt
