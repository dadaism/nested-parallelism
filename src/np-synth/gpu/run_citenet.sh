#!/bin/bash
TYPE=SIMPLE

BIN=./${TYPE}/BLOCK64/gpu-np-synth
#DATA_FILE=../../../data/coPapersCiteseer.graph
#DATA_FILE=../sns_like.txt
DATA_FILE=../cite_network.txt
#DATA_FILE=../../../utility/np-data-generator/dataset1/sns_like_1_2000_4M_0_8.txt
LOG_FILE=np_performance.log

for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 0 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>>  ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
done

TYPE=ARITH
BIN=./${TYPE}/BLOCK64/gpu-np-synth
DATA_FILE=../cite_network.txt
#DATA_FILE=../../../utility/np-data-generator/dataset1/sns_like_1_2000_4M_0_8.txt

for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 0 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>>  ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
done


TYPE=MIX
BIN=./${TYPE}/BLOCK64/gpu-np-synth
DATA_FILE=../cite_network.txt
#DATA_FILE=../../../utility/np-data-generator/dataset1/sns_like_1_2000_4M_0_8.txt

for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 0 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>>  ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
done

TYPE=IO
BIN=./${TYPE}/BLOCK64/gpu-np-synth
DATA_FILE=../cite_network.txt
#DATA_FILE=../../../utility/np-data-generator/dataset1/sns_like_1_2000_4M_0_8.txt

for i in 16	32 64 128 256 512 1024
do
	EXE=$BIN-t$i
	echo "Threshold: $i"  >> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 0 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 1 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 2 &>> ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 3 &>> ${LOG_FILE}
#	$EXE -f 2 -i ${DATA_FILE} -v -s 4 &>>  ${LOG_FILE}
	$EXE -f 2 -i ${DATA_FILE} -v -s 5 &>> ${LOG_FILE}
done


