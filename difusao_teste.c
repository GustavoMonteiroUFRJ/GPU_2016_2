/* Aluno.: Gustavo Ribeir Monteiro */
/* Codigo:  */

/* Para compilar: gcc -o difusao.out difusao.c */
/* Executar contando tempo: time ./difusao.out */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

typedef struct {
	float temp;
	float n;
	float s;
	float e;
	float o;
	float w;
} Node;

int interacao;

Node *v;
float *a;
float *b;

float h1 = 0;
float h2 = 0;
int n1 = 1;
int n2 = 1;

float uo = 0;
float ue = 10;
float us = 5;
float un = 5;

float f_a(int i, int j, int tam) {
	float resp = a[i * tam + j];

	return resp;
}
float f_b(int i, int j, int tam) {
	float resp = b[i * tam + j];

	return resp;
}
float f_o(int i, int j, int tam) {
	float resp = ( 2 + h1 * f_a(i, j, tam) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_e(int i, int j, int tam) {
	float resp = ( 2 - h1 * f_a(i, j, tam) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_s(int i, int j, int tam) {
	float resp = ( 2 + h2 * f_b(i, j, tam) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_n(int i, int j, int tam) {
	float resp = ( 2 - h2 * f_b(i, j, tam) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}

float f_q(int i, int j, int tam) {
	int index = i*tam +j;
	float resp = 2 * (sqrt( v[index].e * v[index].o) * cos(h1 * M_PI) + sqrt(v[index].s * v[index].n) * cos(h2 * M_PI));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_w(int i, int j, int tam) {
	float resp = 2.0 / (1.0 + sqrt(i - pow(f_q(i, j, tam), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_v(int i, int j, int tam) {
	float resp = v[i * tam + j].temp;
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}
void plot_v() {
	int i, j;
	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			printf("%6.3f ", v[i * (n1 + 2) + j].temp );
		}
		printf("\n");
	}
}

void gauss_seidel_sequencial() {
	int tam = n1 + 2;
	int i, j, k;
	float aux;
	Node* pt;

	for ( k = 0; k < interacao; ++k) {
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 1 : 2 ; j < n2 + 1; j += 2) {
				pt = &v[i * tam + j];

				aux = 
				pt->o * f_v(i - 1, j,tam) +
				pt->e * f_v(i + 1, j,tam) +
				pt->s * f_v(i, j - 1,tam) +
				pt->n * f_v(i, j + 1,tam);
				pt->temp = (1 - pt->w) * pt->temp + pt->w * aux;
			}
		}
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 2 : 1; j < n2 + 1; j += 2) {
				pt = &v[i * tam + j];
				
				aux = 
				pt->o * f_v(i - 1, j,tam) +
				pt->e * f_v(i + 1, j,tam) +
				pt->s * f_v(i, j - 1,tam) +
				pt->n * f_v(i, j + 1,tam);
				pt->temp = (1 - pt->w) * pt->temp + pt->w * aux;
			}
		}
	}
	plot_v();
}

void init() {
	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);

	v = (Node*) malloc((n1 + 2) * (n2 + 2) * sizeof(Node));
	a = (float*) malloc((n1 + 2) * (n2 + 2) * sizeof(float));
	b = (float*) malloc((n1 + 2) * (n2 + 2) * sizeof(float));

	if (v == NULL) {
		printf("---Erro de malloc");
		exit(EXIT_FAILURE);
	}

	float x, y;
	int i, j;
	int tam = n1 + 2;
	int temp_media = (ue + uo + un + us) / 4 ;

	Node* pt;
  	// casos de borda
	for ( i = 0; i < n1 + 2; i++ ) {
		v[i * tam + 0].temp = un;
		v[i * tam + n2 + 1].temp = us;
	}
	for ( j = 0; j < n2 + 2; j++ ) {
		v[0 * tam + j].temp = ue;
		v[(n1 + 1) * tam + j].temp = uo;
	}

	// inicializando a e b
	for ( i = 1; i < n1 + 1; i++ ) {
		for ( j = 1; j < n2 + 1; j++ ) {
			x = i * h1;
			y = j * h2;
			a[i * tam + j] = 5.0 * x * (1 - x) * (-y + 0.5);
			b[i * tam + j] = 5.0 * y * (1 - y) * (x - 0.5);
		}
	}

	// inicializando as estruturas
	for ( i = 1; i < n1 + 1; i++ ) {
		for ( j = 1; j < n2 + 1; j++ ) {
			pt = &v[i * tam + j];
			pt->temp = temp_media;
			pt->n = f_n(i,j,tam);
			pt->s = f_s(i,j,tam);
			pt->e = f_e(i,j,tam);
			pt->o = f_o(i,j,tam);
		} 
	}
	// calculando w
	for ( i = 1; i < n1 + 1; i++ ) {
		for ( j = 1; j < n2 + 1; j++ ) {
			v[i * tam + j].w = f_w(i,j,tam);	
		}
	}
	free(a);
	free(b);	
}

int main(int argc, char** argv) {

	FILE *file;
	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);
	interacao = atoi(argv[3]);

	// printf("comecando init\n");
	init();
	// printf("comecando a calcular\n");
	gauss_seidel_sequencial();

	// impressao em arquivo texto
	file = fopen("out.txt", "w");
	int i, j;
	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			fprintf(file, "%6.3f ", v[i * (n1 + 2) + j].temp );
		}
		fprintf(file, "\n");
	}

	fclose(file);
	free(v);

	return 0;
}
