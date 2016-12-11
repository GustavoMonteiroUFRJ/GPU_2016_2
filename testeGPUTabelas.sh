#!/bin/bash

echo "Testing iterations"

x=1
while [ $x -le 5 ]
do
    points=$(( 50 ))
    iterations=$(( $x * 2000 ))
    results="$(./gpu_difusao_tabelas.out $points $points $iterations)"
    echo $results
    printf '\n'
    x=$(( $x + 1 ))
done

echo "Testing points"

x=1
while [ $x -le 5 ]
do
    points=$(( $x * 300 ))
    iterations=$(( 100 ))
    results="$(./gpu_difusao_tabelas.out $points $points $iterations)"
    echo $results
    printf '\n'
    x=$(( $x + 1 ))
done

echo "Testing points and iterations"

x=1
while [ $x -le 5 ]
do
    points=$(( $x * 150 ))
    iterations=$(( $x * 1000 ))
    results="$(./gpu_difusao_tabelas.out $points $points $iterations)"
    echo $results
    printf '\n'
    x=$(( $x + 1 ))
done
