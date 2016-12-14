/* Aluno.: Gustavo Ribeir Monteiro */
/* Codigo:  */

/* Para compilar: nvcc -o gpu_difusao.out gpu_difusao.cu */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

// para tomada de tempo
#include "clock_timer.h"
// GET_TIME(inicio);
// GET_TIME(fim);


#define MAX_THREAD 512
#define MAX_BLOCOS 2000
#define THREADS 16
#define THREAD_X 16
#define THREAD_Y 16

#define FATOR 500

#define M_PIf 3.141592653589f


//para checar erros chamadas Cuda
#define CUDA_SAFE_CALL(call) { \
cudaError_t err = call;     \
if(err != cudaSuccess) {    \
	fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
	exit(EXIT_FAILURE); \
} }

// flags para o programa!
int verboso = 0;
int executa_gpu = 1;
int executa_cpu = 1;

int iteracoes;

int N_threads_x = THREAD_X;
int N_threads_y = THREAD_Y;
int N_blocs_x = 1;
int N_blocs_y = 1;

// variaveis para contar tempo
double inicio, fim;
double tempo_seq, tempo_gpu;
double tempo_init_seq, tempo_ida, tempo_volta;
float delta_eventos;

// matriz da chapa
double *v;

double h1 = 0; // distancia orizontal dos potos
double h2 = 0; // distancia vertical dos potos
int n1 = 1;   // quantidade de potnos em uma linha orizontal
int n2 = 1;   // quantidade de potnos em uma linha vertical

double uo = 0;  // temperatura fixa a oeste
double ue = 10; // temperatura fixa a este
double us = 5;  // temperatura fixa a sul
double un = 5;  // temperatura fixa a norte


// ------------------------- funções de contas cpu -------------------------- //
double f_a(int i, int j) {
	double x = i * h1;
	double y = j * h2;
	double resp = FATOR * x * (1 - x) * (-y + 0.5);
	return resp;
}
double f_b(int i, int j) {
	double x = i * h1;
	double y = j * h2;
	double resp = FATOR * y * (1 - y) * (x - 0.5);
	return resp;
}
double f_o(int i, int j) {
	double resp = ( 2 + h1 * f_a(i, j) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_e(int i, int j) {
	double resp = ( 2 - h1 * f_a(i, j) ) / (4 * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_s(int i, int j) {
	double resp = ( 2 + h2 * f_b(i, j) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_n(int i, int j) {
	double resp = ( 2 - h2 * f_b(i, j) ) / (4 * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_q(int i, int j) {
	double resp = 2 * (sqrt(f_e(i, j) * f_o(i, j)) * cos(h1 * M_PI) + sqrt(f_s(i, j) * f_n(i, j)) * cos(h2 * M_PI));
	// printf("%f\t", sqrt(f_e(i, j) * f_o(i, j)));
	// printf("%f\t", cos(h1 * M_PI));
	// printf("%f\t", sqrt(f_s(i, j) * f_n(i, j)));
	// printf("%f\t", cos(h2 * M_PI));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_w(int i, int j) {
	double resp = 2.0 / (1.0 + sqrt(1 - pow(f_q(i, j), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_v(int i, int j) {
	double resp = v[i * (n1 + 2) + j];
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}

// -------------------------- funções de contas gpu -------------------------- //
__device__ double f_a(int i, int j, double h1, double h2) {
	double x = i * h1;
	double y = j * h2;
	double resp = FATOR * x * (1 - x) * (0.5f - y);
	// printf("x:%f\n", x);
	// printf("y:%f\n", y);
	// printf("h1:%f\n", h1);
	// printf("h2:%f\n", h2);
	// printf("f_a:%f\n", resp);
	return resp;
}
__device__ double f_b(int i, int j, double h1, double h2) {
	double x = i * h1;
	double y = j * h2;
	double resp = FATOR * y * (1 - y) * (x - 0.5f);
	// printf("x:%f\n", x);
	// printf("y:%f\n", y);
	// printf("h1:%f\n", h1);
	// printf("h2:%f\n", h2);
	// printf("f_b:%f\n", resp);
	return resp;
}
__device__ double f_o(int i, int j, double h1, double h2) {
	double resp = ( 2 + h1 * f_a(i, j, h1, h2) ) / (4 * (1 + (h1 * h1) / (h2 * h2)));
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_e(int i, int j, double h1, double h2) {
	double resp = ( 2 - h1 * f_a(i, j, h1, h2) ) / (4 * (1 + (h1 * h1) / (h2 * h2)));
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_s(int i, int j, double h1, double h2) {
	double resp = ( 2 + h2 * f_b(i, j, h1, h2) ) / (4 * (1 + (h2 * h2) / (h1 * h1)));
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_n(int i, int j, double h1, double h2) {
	double resp = ( 2 - h2 * f_b(i, j, h1, h2) ) / (4 * (1 + (h2 * h2) / (h1 * h1)));
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_q(int i, int j, double h1, double h2) {
	double resp = 2 * (sqrt(f_e(i, j, h1, h2) * f_o(i, j, h1, h2)) * cos(h1 * M_PIf) + sqrtf(f_s(i, j, h1, h2) * f_n(i, j, h1, h2)) * cosf(h2 * M_PIf));
	// printf("sqrt(e * o) = %f\n", sqrt(f_e(i, j, h1, h2) * f_o(i, j, h1, h2)));
	// printf("cos(h1 * pi) = %f\n", cos(h1 * M_PI));
	// printf("sqrt(s * n) = %f\n", sqrt(f_s(i, j, h1, h2) * f_n(i, j, h1, h2)));
	// printf("cos(h2 * pi) = %f\n", cos(h2 * M_PI));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_w(int i, int j, double h1, double h2) {
	double resp = 2.0f / (1.0f + sqrtf(1 - powf(f_q(i, j, h1, h2), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_v(double* d_v, int i, int j, int n1) {
	double resp = d_v[i * (n1 + 2) + j];
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}

__device__ void plot_v(double* d_v, int n1, int n2) {
	int i, j;
		for ( i = 0; i < n1 + 2; i++ ) {
			for ( j = 0; j < n2 + 2; j++ ) {
				printf("[%d, %d] = %6.3f ", i, j, d_v[i * (n1 + 2) + j] );
			}
			printf("\n");
		}
		printf("\n");
}

// interação de gauss_seidel em uas etapas
__global__ void kernel_vemelho(double* d_v, double h1, double h2, double n1, double n2, int tam, int verboso) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	double w, aux;

	if (i > 0 && i < n1 + 1 && j > 0 && j < n2 + 1 && (i + j) % 2 == 0) {
		// if (verboso) printf("verm\t[%d][%d]\n", i, j);
		w = f_w(i, j, h1, h2);
		// if (verboso) printf("\tw(%d,%d):%f\n", i, j, w);
		aux =
			f_o(i, j, h1, h2) * f_v(d_v, i - 1, j, n1) +
			f_e(i, j, h1, h2) * f_v(d_v, i + 1, j, n1) +
			f_s(i, j, h1, h2) * f_v(d_v, i, j - 1, n1) +
			f_n(i, j, h1, h2) * f_v(d_v, i, j + 1, n1);
		// if (verboso) printf("\taux(%d,%d):%f\n", i, j, aux);
		// if (verboso) printf("\td_v(%d,%d):%f\n", i, j, f_v(d_v, i, j, n1));
		d_v[i * tam + j] = (1 - w) * f_v(d_v, i, j, n1) + w * aux;
		// if (verboso) printf("\tres(%d,%d):%f\n", i, j, d_v[i * tam + j]);
	}
}
__global__ void kernel_azul(double* d_v, double h1, double h2, double n1, double n2, int tam, int verboso) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	double w, aux;

	//plot_v(d_v, n1, n2);

	if (i > 0 && i < n1 + 1 && j > 0 && j < n2 + 1 && (i + j) % 2 == 1) {
		// if (verboso) printf("verboso %d", verboso);
		// if (verboso) printf("azul\t[%d][%d]\n", i, j);
		w = f_w(i, j, h1, h2);
		// if (verboso) printf("\tw(%d,%d):%f\n", i, j, w);
		aux =
			f_o(i, j, h1, h2) * f_v(d_v, i - 1, j, n1) +
			f_e(i, j, h1, h2) * f_v(d_v, i + 1, j, n1) +
			f_s(i, j, h1, h2) * f_v(d_v, i, j - 1, n1) +
			f_n(i, j, h1, h2) * f_v(d_v, i, j + 1, n1);
		// if (verboso) printf("\taux(%d,%d):%f\n", i, j, aux);
		// if (verboso) printf("\td_v(%d,%d):%f\n", i, j, f_v(d_v, i, j, n1));
		d_v[i * tam + j] = (1 - w) * f_v(d_v, i, j, n1) + w * aux;
		// if (verboso) printf("\tres(%d,%d):%f\n", i, j, d_v[i * tam + j]);
	}
}

// imprime o momento da chapa
void plot_v() {
	int i, j;
	for ( i = 1; i < n1 + 1; i++ ) {
		for ( j = 1; j < n2 + 1; j++ ) {
			printf("%6.3f ", v[i * (n1 + 2) + j] );
		}
		printf("\n");
	}
	printf("\n");
}

// funcão que trata tudo que for necessario para executar em gpu
void gauss_seidel_sequencial_gpu() {
	int tam = n1 + 2;
	double* d_v;
	int k;
	cudaEvent_t start, stop;

	GET_TIME(inicio);
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_v, (n1 + 2) * (n2 + 2) * sizeof(double)));
	CUDA_SAFE_CALL(cudaMemcpy(d_v, v, (n1 + 2) * (n2 + 2) * sizeof(double), cudaMemcpyHostToDevice));
	GET_TIME(fim);

	tempo_ida = fim - inicio;

	N_blocs_x = (n1 + 2) / THREAD_X + 1;
	N_blocs_y = (n2 + 2)/ THREAD_Y + 1;

	dim3 threads_per_blocos(N_threads_x, N_threads_y);
	dim3 grade_blocos(N_blocs_x , N_blocs_y);

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start));

	for ( k = 0; k < iteracoes; ++k) {
		kernel_vemelho <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam, verboso);
		CUDA_SAFE_CALL(cudaGetLastError());

		kernel_azul <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam, verboso);
		CUDA_SAFE_CALL(cudaGetLastError());
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

	GET_TIME(inicio);
	CUDA_SAFE_CALL(cudaMemcpy( v, d_v, (n1 + 2) * (n2 + 2) * sizeof(double), cudaMemcpyDeviceToHost));
	GET_TIME(fim);

	tempo_volta =  fim - inicio;
}

// inicialização das variáveis globais e da matriz
void init() {
	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);

	v = (double*) malloc((n1 + 2) * (n2 + 2) * sizeof(double));

	if (v == NULL) {
		printf("---Erro de malloc");
		exit(EXIT_FAILURE);
	}

	int i, j;
	int tam = n1 + 2;

	for ( i = 0; i < n1 + 2; i++ ) {
		for ( j = 0; j < n2 + 2; j++ ) {
			if (0) ; // so para deixar bonitinho
			else if (j == 0) 		v[i * tam + j] = uo;
			else if (j == n2 + 1) 	v[i * tam + j] = ue;
			else if (i == 0) 		v[i * tam + j] = us;
			else if (i == n1 + 1) 	v[i * tam + j] = un;
			else v[i * tam + j] = (ue + uo + un + us) / 4;
		}
	}
}

void gauss_seidel_sequencial() {
	int tam = n1 + 2;
	int i, j, k;
	double aux;

	for ( k = 0; k < iteracoes; ++k) {
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 1 : 2 ; j < n2 + 1; j += 2) {
				// if (verboso) printf("verm\t[%d][%d]\n", i, j);
				// if (verboso) printf("\tw:%f\n", f_w(i, j));
				aux =
					f_o(i, j) * f_v(i - 1, j) +
					f_e(i, j) * f_v(i + 1, j) +
					f_s(i, j) * f_v(i, j - 1) +
					f_n(i, j) * f_v(i, j + 1);
				// if (verboso) printf("\taux:%f\n", aux);

				v[i * tam + j] = (1 - f_w(i, j)) * f_v(i, j) + f_w(i, j) * aux;
				// if (verboso) printf("\td_v:%f\n", v[i * tam + j]);
			}
		}
		for ( i = 1; i < n1 + 1; i += 1) {
			for ( j = i % 2 ? 2 : 1; j < n2 + 1; j += 2) {
				// if (verboso) printf("azul\t[%d][%d]\n", i, j);
				// if (verboso) printf("\tw:%f\n", f_w(i, j));
				aux =
					f_o(i, j) * f_v(i - 1, j) +
					f_e(i, j) * f_v(i + 1, j) +
					f_s(i, j) * f_v(i, j - 1) +
					f_n(i, j) * f_v(i, j + 1);
				// if (verboso) printf("\taux:%f\n", aux);

				v[i * tam + j] = (1 - f_w(i, j)) * f_v(i, j) + f_w(i, j) * aux;
				// if (verboso) printf("\td_v:%f\n", v[i * tam + j]);
			}
		}
	}
}

// checa se a entrada da main está correta e incicializa as falgs
int checa_entrada(const int argc, const char** argv){
	if( argc < 3){
		printf("Falta de parametros!!! Entre com %s <numero de pontos em x> <numero de pontos em y> <numero de interações>\n", argv[0]);
		return -1;
	}
	if (argv[1][1] == '-' || argv[2][1] == '-' || argv[3][1] == '-'){
		return -1;
	}
	if( argc > 3){
		for (int i = 0; i < argc; ++i){
			if(argv[i][0] == '-'){
				switch (argv[i][1]){
					case 'v':
					case 'V':
					verboso = 1;
					break;

					case 'g':
					case 'G':
					executa_cpu = 0;
					break;

					case 'c':
					case 'C':
					executa_gpu = 0;
					break;
				}
			}
		}
	}
	return 0;
}

int main(const int argc, const char** argv) {

	if(checa_entrada(argc,argv) == -1) return -1;

	FILE *file;
	n1 = atoi(argv[1]);
	n2 = atoi(argv[2]);
	iteracoes = atoi(argv[3]);

	/*-------------- GPU -----------------*/
	if(executa_gpu){
		if(verboso) printf("Iniciando GPU...\n");
		init();

		if(verboso) plot_v();
		if(verboso) printf(" Executando\n");
		gauss_seidel_sequencial_gpu();

		tempo_gpu = fim - inicio;

		// impressao em arquivo texto
		file = fopen("out_gpu.txt", "w");
		for ( int i = 0; i < n1 + 2; i++ ) {
			for ( int j = 0; j < n2 + 2; j++ ) {
				fprintf(file, "%6.3f ", v[i * (n1 + 2) + j] );
			}
			fprintf(file, "\n");
		}
		if(verboso) plot_v();

		fclose(file);
		free(v);
	}

	/*-------------- CPU -----------------*/
	if( executa_cpu){

		if(verboso) printf("Iniciando CPU...\n");
		GET_TIME(inicio);
		init();
		GET_TIME(fim);
		tempo_init_seq = fim - inicio;

		if(verboso) plot_v();
		if(verboso) printf(" Executando\n");
		GET_TIME(inicio);
		gauss_seidel_sequencial();
		GET_TIME(fim);

		tempo_seq = fim - inicio;

		// impressao em arquivo texto
		file = fopen("out_cpu.txt", "w");
		for ( int i = 0; i < n1 + 2; i++ ) {
			for ( int j = 0; j < n2 + 2; j++ ) {
				fprintf(file, "%6.3f ", v[i * (n1 + 2) + j] );
			}
			fprintf(file, "\n");
		}
		if(verboso) plot_v();

		fclose(file);
		free(v);
	}

	//------------------------------- imprime dos tempos de execucao ----------------------//
	printf("Threads: %d\n", THREADS);
	printf("Pontos: %d x %d\n", n1, n2);
	printf("Iteracões: %d\n", iteracoes);
	printf("Tempo total sequencial          = %f seg \n", tempo_seq + tempo_init_seq);
	printf("Tempo total GPU                 = %f seg \n\n", delta_eventos / 1000 + tempo_ida + tempo_volta);
	printf("Tempo trabalho sequencial       = %f seg \n", tempo_seq);
	printf("Tempo trabalho paralelo         = %f seg \n\n", delta_eventos / 1000);
	printf("Tempo inicialização sequencial  = %f seg \n", tempo_init_seq);
	printf("Tempo inicialização da gpu 	= %f seg \n", tempo_ida);
	printf("Tempo volta da gpu 	= %f seg \n", tempo_volta);

	CUDA_SAFE_CALL(cudaDeviceReset());

	return 0;
}
