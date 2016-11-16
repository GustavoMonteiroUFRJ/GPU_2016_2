/* Aluno.: Gustavo Ribeir Monteiro */
/* Codigo:  */

/* Para compilar: nvcc -o difusao.out difusao.cu */


/*

LISTA DA COISAS PARA FAZER

* implementar as coisas para GPU
* implementar a main

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

// para tomada de tempo
// #include "clock_timer.h"
// GET_TIME(inicio);
// GET_TIME(fim);

#define INTERACAO 2

#define MAX_THREAD 512
#define MAX_BLOCOS 2000
#define THREAD_X 32
#define THREAD_Y 16

/*
//para checar erros chamadas Cuda
#define CUDA_SAFE_CALL(call) { \
   cudaError_t err = call;     \
   if(err != cudaSuccess) {    \
      fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); } }
*/
/*
int N_threads_x = THREAD_X;
int N_threads_y = THREAD_Y;
int N_blocs_x = 1;
int N_blocs_y = 1;

double inicio, fim;
double tempo_seq, tempo_gpu;
double tempo_ida, tempo_volta;
float delta_eventos;
*/

float *v;

float h1 = 0;
float h2 = 0;
int n1 = 1;
int n2 = 1;

float uo = 0;
float ue = 10;
float us = 5;
float un = 5;


float f_a(int i, int j) {
	float x = i * h1;
	float y = j * h2;
	float resp = 500 * x * (1 - x) * (-y + 0.5);
	return resp;
}
float f_b(int i, int j) {
	float x = i * h1;
	float y = j * h2;
	float resp = 500 * y * (1 - y) * (x - 0.5);
	return resp;
}

float f_o(int i, int j) {
	float resp = ( 2 + h1 * f_a(i, j) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	printf("f_o(%d,%d) = %f\n", i, j, resp);
	return;
}
float f_e(int i, int j) {
	float resp = ( 2 - h1 * f_a(i, j) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_s(int i, int j) {
	float resp = ( 2 + h2 * f_b(i, j) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_n(int i, int j) {
	float resp = ( 2 - h2 * f_b(i, j) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_q(int i, int j) {
	float resp = 2 * (sqrt(f_e(i, j) * f_o(i, j)) * cos(h1 * M_PI) + sqrt(f_s(i, j) * f_n(i, j)) * cos(h2 * M_PI));
	printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_w(int i, int j) {
	float resp = 2.0 / (1.0 + sqrt(i - pow(f_q(i, j), 2)));
	printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}

float f_v(int i, int j) {
	float resp = v[i * (n1 + 2) + j];
	printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}

void plot_v() {
	int i, j;
	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			printf("%0.4f ", v[i * (n1 + 2) + j] );
		}
		printf("\n");
	}
}

void gauss_seidel_sequencial() {
	int tam = n1 + 2;
	int i, j, k;
	float aux;

	for ( k = 0; k < INTERACAO; ++k) {
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 1 : 2 ; j < n2 + 1; j += 2) {
				// sleep(1);
				printf("[%d,%d] \n", i, j );
				aux = f_o(i, j) * f_v(i - 1, j) +
				      f_e(i, j) * f_v(i + 1, j) +
				      f_s(i, j) * f_v(i, j - 1) +
				      f_n(i, j) * f_v(i, j + 1);
				v[i * tam + j] = (1 - f_w(i, j)) * f_v(i, j) + f_w(i, j) * aux;
			}
		}
		printf("\n");
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 2 : 1; j < n2 + 1; j += 2) {
				printf("[%d,%d] \n", i, j );
				// sleep(1);

				aux = f_o(i, j) * f_v(i - 1, j) +
				      f_e(i, j) * f_v(i + 1, j) +
				      f_s(i, j) * f_v(i, j - 1) +
				      f_n(i, j) * f_v(i, j + 1);
				v[i * tam + j] = (1 - f_w(i, j)) * f_v(i, j) + f_w(i, j) * aux;
			}
		}
		printf("\n");
		plot_v();
	}
}

void init() {
	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);

	v = (float*) malloc((n1 + 2) * (n2 + 2) * sizeof(float));

	if (v == NULL) {
		printf("---Erro de malloc");
		exit(EXIT_FAILURE);
	}

	int i, j;
	int tam = n1 + 2;

	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			if (0) printf("so para deixar bonitinho");
			else if (i == 0) 		v[i * tam + j] = ue;
			else if (i == n1 + 1) 	v[i * tam + j] = uo;
			else if (j == 0) 		v[i * tam + j] = un;
			else if (j == n2 + 1) 	v[i * tam + j] = us;
			else v[i * tam + j] = (ue + uo + un + us) / 4;
		}
	}
}

/*
__global__ void kernel_linear () {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
}
__global__ void kernel_binimencional() {
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
}
int funcao_gpu() {

	float *d_memoria_GPU;
	cudaEvent_t start, stop;

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_memoria_GPU, bytes ));
	CUDA_SAFE_CALL(cudaMemcpy(d_memoria_GPU, h_memoria_CPU, bytes, cudaMemcpyHostToDevice));

	int n_threads;
	int n_blocos;
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start));

	n_threads = 1;
	n_blocos = 1;

	kernel_linear <<< n_blocos , MAX_THREAD>>> ();

	N_blocs_x = 1;
	N_blocs_y = 1;

	dim3 threads_per_blocos(N_threads_x, N_threads_y);
	dim3 grade_blocos(N_blocs_x , N_blocs_y);

	kernel_binimencional <<< grade_blocos , threads_per_blocos >>> ();

	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaEventRecord(stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

	CUDA_SAFE_CALL(cudaMemcpy( d_memoria_CPU, h_memoria_GPU, bytes, cudaMemcpyDeviceToHost));

	return 0;
}
*/



int main(int argc, char** argv) {

	FILE *file;
	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);

	printf("comecando init\n");
	init();
	printf("comecando a calcular\n");
	gauss_seidel_sequencial();

	// impressao em arquivo texto
	file = fopen("out.txt", "w");
	int i, j;
	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			fprintf(file, "%0.4f ", v[i * (n1 + 2) + j] );
		}
		fprintf(file, "\n");
	}

	fclose(file);
	free(v);

	return 0;
}
