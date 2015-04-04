#!/bin/bash

BIN=./BLOCK64/gpu-bc
#DATA_FILE=../../../data/p2p-Gnutella31.txt
DATA_FILE=../../../utility/graph-generator/graph_5000_1200_0_4.gr
#DATA_FILE=data_5000.gr
LOG_FILE=bc_performance.log

for i in 16	32 64 128 256 512 1024
#for i in 1024 512 256 128 64 32 16
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 6 &>> ${LOG_FILE}
done

DATA_FILE=../../../utility/graph-generator/graph_5000_1200_0_6.gr
for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 6 &>> ${LOG_FILE}
done

DATA_FILE=../../../utility/graph-generator/graph_5000_1200_0_8.gr
for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 6 &>> ${LOG_FILE}
done

DATA_FILE=../../../utility/graph-generator/graph_5000_1200_1_2.gr
for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 6 &>> ${LOG_FILE}
done

DATA_FILE=../../../utility/graph-generator/graph_5000_1200_1_4.gr
for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 6 &>> ${LOG_FILE}
done
