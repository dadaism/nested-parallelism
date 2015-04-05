#!/bin/bash

BIN=./cuSparse
#DATA_FILE=../../../datasets/DIMACS10/coPapersCiteseer.graph
DATA_FILE=../../../datasets/SLNDC/soc-LiveJournal1.txt 
LOG_FILE=cusparse_performance.log

EXE=$BIN

$EXE -f 2 -i ${DATA_FILE} -v  #&>> ${LOG_FILE}
