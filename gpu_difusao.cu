/* Aluno.: Gustavo Ribeir Monteiro */
/* Codigo:  */

/* Para compilar: nvcc -o difusao.out difusao.cu */


/*

LISTA DA COISAS PARA FAZER

* implementar as coisas para GPU
* implementar a main
* use_fest_math flag para compilador! ( pesquisar )

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

// para tomada de tempo
#include "clock_timer.h"
// GET_TIME(inicio);
// GET_TIME(fim);

int interacoes;

#define MAX_THREAD 512
#define MAX_BLOCOS 2000
#define THREAD_X 16
#define THREAD_Y 16

#define M_PIf 3.141592653589f



//para checar erros chamadas Cuda
#define CUDA_SAFE_CALL(call) { \
cudaError_t err = call;     \
if(err != cudaSuccess) {    \
	fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
	exit(EXIT_FAILURE); } }

	int N_threads_x = THREAD_X;
	int N_threads_y = THREAD_Y;
	int N_blocs_x = 1;
	int N_blocs_y = 1;

// variaveis para contar tempo
	double inicio, fim;
	double tempo_seq, tempo_gpu;
	double tempo_ida, tempo_volta;
	float delta_eventos;

// matriz da chapa
	float *v;

float h1 = 0; // distancia orizontal dos potos
float h2 = 0; // distancia vertical dos potos
int n1 = 1;   // quantidade de potnos em uma linha orizontal
int n2 = 1;   // quantidade de potnos em uma linha vertical

float uo = 0;  // temperatura fixa a oeste
float ue = 10; // temperatura fixa a este
float us = 5;  // temperatura fixa a sul
float un = 5;  // temperatura fixa a norte


// ------------------------- funções de contas cpu -------------------------- //
float f_a(int i, int j) {
	float x = i * h1;
	float y = j * h2;
	float resp = 5.0 * x * (1 - x) * (-y + 0.5);
	return resp;
}
float f_b(int i, int j) {
	float x = i * h1;
	float y = j * h2;
	float resp = 5.0 * y * (1 - y) * (x - 0.5);
	return resp;
}
float f_o(int i, int j) {
	float resp = ( 2 + h1 * f_a(i, j) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_e(int i, int j) {
	float resp = ( 2 - h1 * f_a(i, j) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_s(int i, int j) {
	float resp = ( 2 + h2 * f_b(i, j) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_n(int i, int j) {
	float resp = ( 2 - h2 * f_b(i, j) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_q(int i, int j) {
	float resp = 2 * (sqrt(f_e(i, j) * f_o(i, j)) * cos(h1 * M_PI) + sqrt(f_s(i, j) * f_n(i, j)) * cos(h2 * M_PI));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_w(int i, int j) {
	float resp = 2.0 / (1.0 + sqrt(i - pow(f_q(i, j), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_v(int i, int j) {
	float resp = v[i * (n1 + 2) + j];
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}


// -------------------------- funções de contas gpu -------------------------- //
__device__ float f_a(int i, int j, float h1, float h2) {
	float x = i * h1;
	float y = j * h2;
	float resp = 500 * x * (1 - x) * (-y + 0.5f);
	return resp;
}
__device__ float f_b(int i, int j, float h1, float h2) {
	float x = i * h1;
	float y = j * h2;
	float resp = 500 * y * (1 - y) * (x - 0.5f);
	return resp;
}
__device__ float f_o(int i, int j, float h1, float h2) {
	float resp = ( 2 + h1 * f_a(i, j, h1, h2) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_e(int i, int j, float h1, float h2) {
	float resp = ( 2 - h1 * f_a(i, j, h1, h2) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_s(int i, int j, float h1, float h2) {
	float resp = ( 2 + h2 * f_b(i, j, h1, h2) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_n(int i, int j, float h1, float h2) {
	float resp = ( 2 - h2 * f_b(i, j, h1, h2) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_q(int i, int j, float h1, float h2) {
	float resp = 2 * (sqrtf(f_e(i, j, h1, h2) * f_o(i, j, h1, h2)) * cosf(h1 * M_PIf) + sqrtf(f_s(i, j, h1, h2) * f_n(i, j, h1, h2)) * cosf(h2 * M_PIf));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_w(int i, int j, float h1, float h2) {
	float resp = 2.0f / (1.0f + sqrtf(i - powf(f_q(i, j, h1, h2), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_v(float* d_v, int i, int j, int n1) {
	float resp = d_v[i * (n1 + 2) + j];
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}

// interação de gauss_seidel em uas etapas
__global__ void kernel_vemelho(float* d_v, float h1, float h2, float n1, float n2, int tam) {
	int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int j = 1 + 2 * blockIdx.y * blockDim.y + threadIdx.y;
	j += i % 2;

	if (i < n1 + 1 && j < n2 + 1) {
		float w = f_w(i, j, h1, h2);
		d_v[i * tam + j] = (1 - w) * f_v(d_v, i, j, h1) + w *
		(f_o(i, j, h1, h2) * f_v(d_v, i - 1, j, h1) +
			f_e(i, j, h1, h2) * f_v(d_v, i + 1, j, h1) +
			f_s(i, j, h1, h2) * f_v(d_v, i, j - 1, h1) +
			f_n(i, j, h1, h2) * f_v(d_v, i, j + 1, h1));
	}
}
__global__ void kernel_azul(float* d_v, float h1, float h2, float n1, float n2, int tam) {
	int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int j = 1 + 2 * blockIdx.y * blockDim.y + threadIdx.y;
	j += (i + 1) % 2;

	if (i < n1 + 1 && j < n2 + 1) {
		float w = f_w(i, j, h1, h2);
		d_v[i * tam + j] = (1 - w) * f_v(d_v, i, j, h1) + w *
		(f_o(i, j, h1, h2) * f_v(d_v, i - 1, j, h1) +
			f_e(i, j, h1, h2) * f_v(d_v, i + 1, j, h1) +
			f_s(i, j, h1, h2) * f_v(d_v, i, j - 1, h1) +
			f_n(i, j, h1, h2) * f_v(d_v, i, j + 1, h1));
}
}

// imprime o momento da chapa
void plot_v() {
	int i, j;
	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			printf("%0.4f ", v[i * (n1 + 2) + j] );
		}
		printf("\n");
	}
}

// funcão que trata tudo que for necessario para executar em gpu
void gauss_seidel_sequencial_gpu() {
	int tam = n1 + 2;
	float* d_v;
	int k;
	cudaEvent_t start, stop;

	GET_TIME(inicio);
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_v, (n1 + 2) * (n2 + 2) * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(d_v, v, (n1 + 2) * (n2 + 2) * sizeof(float) , cudaMemcpyHostToDevice));
	GET_TIME(fim);

	tempo_ida = fim - inicio;

	N_blocs_x = (n1 + 2) / THREAD_X + 1;
	N_blocs_y = (n2 + 2)/ THREAD_Y + 1;

	dim3 threads_per_blocos(N_threads_x, N_threads_y);
	dim3 grade_blocos(N_blocs_x , N_blocs_y);

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start));

	for ( k = 0; k < interacoes; ++k) {
		kernel_vemelho <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam);
		CUDA_SAFE_CALL(cudaGetLastError());

		kernel_azul <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam);
		CUDA_SAFE_CALL(cudaGetLastError());
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

	GET_TIME(inicio);
	CUDA_SAFE_CALL(cudaMemcpy( v, d_v, (n1 + 2) * (n2 + 2) * sizeof(float), cudaMemcpyDeviceToHost));
	GET_TIME(fim);

	tempo_volta =  inicio - fim;
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
			if (0) ; // so para deixar bonitinho
			else if (i == 0) 		v[i * tam + j] = ue;
			else if (i == n1 + 1) 	v[i * tam + j] = uo;
			else if (j == 0) 		v[i * tam + j] = un;
			else if (j == n2 + 1) 	v[i * tam + j] = us;
			else v[i * tam + j] = (ue + uo + un + us) / 4;
		}
	}
}

void gauss_seidel_sequencial() {
	int tam = n1 + 2;
	int i, j, k;
	float aux;

	for ( k = 0; k < interacoes; ++k) {
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 1 : 2 ; j < n2 + 1; j += 2) {
				aux = f_o(i, j) * f_v(i - 1, j) +
				      f_e(i, j) * f_v(i + 1, j) +
				      f_s(i, j) * f_v(i, j - 1) +
				      f_n(i, j) * f_v(i, j + 1);
				v[i * tam + j] = (1 - f_w(i, j)) * f_v(i, j) + f_w(i, j) * aux;
			}
		}
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 2 : 1; j < n2 + 1; j += 2) {
				aux = f_o(i, j) * f_v(i - 1, j) +
				      f_e(i, j) * f_v(i + 1, j) +
				      f_s(i, j) * f_v(i, j - 1) +
				      f_n(i, j) * f_v(i, j + 1);
				v[i * tam + j] = (1 - f_w(i, j)) * f_v(i, j) + f_w(i, j) * aux;
			}
		}
	}
}

int main(int argc, char** argv) {

	if( argc < 3){
		printf("Falta de parametros!!! Entre com %s <numero de pontos em x> <numero de pontos em y> <numero de interações>\n", argv[0]);
		return -1;
	}

	FILE *file;
	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);
	interacoes = atoi(argv[3]);

	/*-------------- GPU -----------------*/
	init();

	gauss_seidel_sequencial_gpu();

	// impressao em arquivo texto
	file = fopen("out_gpu.txt", "w");
	int i, j;
	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			fprintf(file, "%6.3f ", v[i * (n1 + 2) + j] );
		}
		fprintf(file, "\n");
	}

	fclose(file);
	free(v);

	/*-------------- CPU -----------------*/

	init();

	GET_TIME(inicio);
	gauss_seidel_sequencial();
	GET_TIME(fim);

	tempo_seq = fim - inicio;

	// impressao em arquivo texto
	file = fopen("out_cpu.txt", "w");
	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			fprintf(file, "%6.3f ", v[i * (n1 + 2) + j] );
		}
		fprintf(file, "\n");
	}

	fclose(file);
	free(v);

	//------------------------------- imprime dos tempos de execucao ----------------------//
	printf("Tempo sequencial = %g seg \n", tempo_seq);
	printf("Tempo trabalho paralelo   = %f seg \n\n", delta_eventos / 1000);
	printf("Tempo total paralelo   = %f seg \n", delta_eventos / 1000 + tempo_ida + tempo_volta);
	printf("Tempo ida (kernel)   = %f seg \n", tempo_ida);
	printf("Tempo volta (kernel) = %f seg \n", tempo_volta);

	CUDA_SAFE_CALL(cudaDeviceReset());

	return 0;
}
