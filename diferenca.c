#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>

#define MODULO(x) ( (x) >= 0 ? (x) : -(x) )

#define Erro 1e-5

int main(int argc, char const *argv[]){
	
	if( argc < 4 ){
		printf("Entre com o nome dos 2 arquivos e o tamanho de colunas\n");
	}
	int flag = 0;

	FILE* arq1 = fopen(argv[1], "r");
	FILE* arq2 = fopen(argv[2], "r");
	int colunas = atoi(argv[3]);

	float f1;
	float f2;
	float dif;
	int cont=0;


	while(!feof(arq1)) {
		fscanf(arq1,"%f,", &f1);
		fscanf(arq2,"%f,", &f2);
		dif = f1 - f2;
		cont++;
		if( MODULO(dif) > Erro ){
			printf(" [%d,%d] %6.3f & %6.3f \n",cont / colunas,cont % colunas,f1,f2);
			flag = 1;
		}
	}
	if (!flag) printf("Tudo Ok! =)\n");

	return 0;
}
