#!/bin/bash

if [ -a "graph" ]
then rm graph
fi

if [ -a "main.o" ]
then rm main.o
fi

if [ -a "wgraph.o" ]
then rm wgraph.o
fi

if [ -a "mygraph_generated_random.txt" ]
then rm mygraph_generated_random.txt
fi

if [ -a "mygraph_generated_random.dot" ]
then rm mygraph_generated_random.dot
fi

if [ -a "mygraph_generated_random.jpeg" ]
then rm mygraph_generated_random.jpeg
fi

if [ -a "mygraph_loaded_random.dot" ]
then rm mygraph_loaded_random.dot
fi

if [ -a "mygraph_loaded_random.jpeg" ]
then rm mygraph_loaded_random.jpeg
fi

if [ -a "mygraph_generated_small_world_lattice.txt" ]
then rm mygraph_generated_small_world_lattice.txt
fi

if [ -a "mygraph_generated_small_world_lattice.dot" ]
then rm mygraph_generated_small_world_lattice.dot
fi

if [ -a "mygraph_generated_small_world_lattice.jpeg" ]
then rm mygraph_generated_small_world_lattice.jpeg
fi

if [ -a "mygraph_generated_small_world_random.txt" ]
then rm mygraph_generated_small_world_random.txt
fi

if [ -a "mygraph_generated_small_world_random.dot" ]
then rm mygraph_generated_small_world_random.dot
fi

if [ -a "mygraph_generated_small_world_random.jpeg" ]
then rm mygraph_generated_small_world_random.jpeg
fi

if [ -a "mygraph_loaded_small_world_lattice.dot" ]
then rm mygraph_loaded_small_world_lattice.dot
fi

if [ -a "mygraph_loaded_small_world_lattice.jpeg" ]
then rm mygraph_loaded_small_world_lattice.jpeg
fi

if [ -a "mygraph_loaded_small_world_random.dot" ]
then rm mygraph_loaded_small_world_random.dot
fi

if [ -a "mygraph_loaded_small_world_random.jpeg" ]
then rm mygraph_loaded_small_world_random.jpeg
fi

if [ -a "mygraph_generated_scale_free.txt" ]
then rm mygraph_generated_scale_free.txt
fi

if [ -a "mygraph_generated_scale_free.dot" ]
then rm mygraph_generated_scale_free.dot
fi

if [ -a "mygraph_generated_scale_free.jpeg" ]
then rm mygraph_generated_scale_free.jpeg
fi

if [ -a "mygraph_loaded_scale_free.dot" ]
then rm mygraph_loaded_scale_free.dot
fi

if [ -a "mygraph_loaded_scale_free.jpeg" ]
then rm mygraph_loaded_scale_free.jpeg
fi

if [ -a "mygraph_generated_random.gr" ]
then rm mygraph_generated_random.gr
fi

if [ -a "mygraph_generated_small_world.gr" ]
then rm mygraph_generated_small_world.gr
fi

if [ -a "mygraph_generated_scale_free.gr" ]
then rm mygraph_generated_scale_free.gr
fi