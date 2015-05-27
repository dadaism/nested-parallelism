#!/bin/bash

BIN=../src/gpu/gpu-graph-color
DATA_FILE=../../../datasets/DIMACS10/coPapersCiteseer.graph
LOG_FILE=sp_performance.log

EXE=$BIN
$EXE -f 1 -i ${DATA_FILE} -v -s 7
