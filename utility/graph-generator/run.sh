#!/bin/bash

#./graph-gen -n 3000 -l 1 -m 1200 -g 0.2 -d 1 -e graph_3000_1200_0_2.gr
./graph-gen -n 5000 -l 1 -m 1200 -g 2.8 -d 1 -e graph_5000_1200_2_8.gr

#for i in 25000
#do
#	./graph-gen -n $i -l 1 -m 8 -d 0 -e graph_${i}_1_8.gr
#	./graph-gen -n $i -l 8 -m 8 -d 0 -e graph_${i}_8_8.gr
#	./graph-gen -n $i -l 32 -m 32 -d 0 -e graph_${i}_32_32.gr
#	./graph-gen -n $i -l 32 -m 256 -d 0 -e graph_${i}_32_256.gr
#	./graph-gen -n $i -l 32 -m 1024 -d 0 -e graph_${i}_32_1024.gr
#	./graph-gen -n $i -l 64 -m 64 -d 0 -e graph_${i}_64_64.gr
#	./graph-gen -n $i -l 64 -m 256 -d 0 -e graph_${i}_64_256.gr
#	./graph-gen -n $i -l 64 -m 1024 -d 0 -e graph_${i}_64_1024.gr
#done

