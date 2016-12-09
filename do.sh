#!/bin/bash


if [ $# -ge 2 ]; then

oredem_da_matriz=$1
interacoes=$2
flag1=$3
flag2=$4

nvcc gpu_difusao_tabelas.cu -o otim

./otim $oredem_da_matriz $oredem_da_matriz $interacoes $flag1 $flag2

	if [ $# -lt 3 ]; then

	 gcc diferenca.c -o compara
	 colunas=$(($oredem_da_matriz + 2))
	 ./compara out_gpu.txt out_cpu.txt $colunas

	fi
else  
	echo "Entre com <oredem da matriz> <interações>"
fi

