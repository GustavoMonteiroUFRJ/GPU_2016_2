#!/bin/bash

if [ $# -ge 2 ]; then

nvcc gpu_difusao_tabelas.cu -o otim

./otim $1 $1 $2  $3

	if [ $# -lt 3 ]; then

	 gcc diferenca.c -o compara

	 ./compara out_gpu.txt out_cpu.txt 32

	fi
else  
	echo "Entre com <oredem da matriz> <interações>"
fi

