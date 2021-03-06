/*

*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

// para tomada de tempo
#include "clock_timer.h"

#define MAX_THREAD 512
#define MAX_BLOCOS 2000
#define THREADS 16

#define M_PIf 3.141592653589f
#define FATOR 500.0
#define FATORf 500.0f

#define PLOT(m,k) {int _i, _j; \
	for ( _i = 1; _i < n2 + 1; ++_i){	\
		for ( _j = 1; _j < n1 + 1; ++_j){\
			printf("%f ", v[_i*(n2+2) + _j].k);\
		}	\
		printf("\n");\
	}\
	printf("\n");\
}
#define CUDA_SAFE_CALL(call) { \
	cudaError_t err = call;     \
	if(err != cudaSuccess) {    \
		fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
}

typedef struct {
	double temp;
	double n;
	double s;
	double e;
	double o;
	double w;
} Node;

// flags para o programa!
int verboso = 0;
int executa_gpu = 1;
int executa_cpu = 1;

int iteracoes;

int N_threads_x = THREADS;
int N_threads_y = THREADS;
int N_blocs_x = 1;
int N_blocs_y = 1;

/* faltam as variavas de tempo*/

// matrizes cpu da chapa
Node *v;
double *a;
double *b;

// matrizes gpu da chapa
Node *d_v;
double *d_a;
double *d_b;

double h1 = 0; // distancia orizontal dos potos
double h2 = 0; // distancia vertical dos potos
double denominadro1; // = (4 * (1 + (h1 * h1 / (h2 * h2))))
double denominadro2; // = (4 * (1 + (h2 * h2 / (h1 * h1))))
int n1 = 1;   // quantidade de potnos em uma linha orizontal
int n2 = 1;   // quantidade de potnos em uma linha vertical

double uo = 0;  // temperatura fixa a oeste
double ue = 10; // temperatura fixa a este
double us = 5;  // temperatura fixa a sul
double un = 5;  // temperatura fixa a norte

// variaveis para contar tempo
double inicio, fim;
double tempo_seq, tempo_gpu;
double time_init_seq, time_init_gpu, tempo_volta;
float delta_eventos;

// ------------------------- funções de contas cpu -------------------------- //
double f_a(int i, int j, int tam) {
	double resp = a[i * tam + j];

	return resp;
}
double f_b(int i, int j, int tam) {
	double resp = b[i * tam + j];

	return resp;
}
double f_o(int i, int j, int tam) {
	double resp = ( 2.0 + h1 * f_a(i, j, tam) ) / denominadro1;
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_e(int i, int j, int tam) {
	double resp = ( 2.0 - h1 * f_a(i, j, tam) ) / denominadro1;
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_s(int i, int j, int tam) {
	double resp = ( 2.0 + h2 * f_b(i, j, tam) ) / denominadro2;
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_n(int i, int j, int tam) {
	double resp = ( 2.0 - h2 * f_b(i, j, tam) ) / denominadro2;
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}

double f_q(int i, int j, int tam) {
	int index = i*tam +j;
	double resp = 2.0 * (sqrt( v[index].e * v[index].o) * cos(h1 * M_PI) + sqrt(v[index].s * v[index].n) * cos(h2 * M_PI));
	// printf("%f\t", sqrt(f_e(i, j, tam) * f_o(i, j, tam)));
	// printf("%f\t", cos(h1 * M_PI));
	// printf("%f\t", sqrt(f_s(i, j, tam) * f_n(i, j, tam)));
	// printf("%f\t", cos(h2 * M_PI));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_w(int i, int j, int tam) {
	double resp = 2.0 / (1.0 + sqrt(1.0 - pow(f_q(i, j, tam), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
double f_v(int i, int j, int tam) {
	double resp = v[i * tam + j].temp;
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}

// -------------------------- funções de contas gpu -------------------------- //
__device__ double f_a(double* d_a, int i, int j, int tam) {
	double resp = d_a[i * tam + j];

	return resp;
}
__device__ double f_b(double* d_b, int i, int j, int tam) {
	double resp = d_b[i * tam + j];

	return resp;
}
__device__ double f_o(double* d_a, int i, int j, int tam, double h1, double d_denominadro1) {
	double resp = ( 2.0f + h1 * f_a(d_a, i, j, tam) ) / d_denominadro1;
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_e(double* d_a, int i, int j,int tam,double h1, double d_denominadro1) {
	double resp = ( 2.0f - h1 * f_a(d_a,i, j, tam) ) / d_denominadro1;
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_s(double* d_b, int i, int j,int tam,double h2, double d_denominadro2) {
	double resp = ( 2.0f + h2 * f_b(d_b, i, j, tam) ) / d_denominadro2;
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_n(double* d_b, int i, int j,int tam,double h2, double d_denominadro2){
	double resp = ( 2.0f - h2 * f_b(d_b, i, j, tam) ) / d_denominadro2;
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}

__device__ double f_q(Node* d_v, int i, int j,int tam, double h1, double h2) {
	int index = i * tam + j;
	double resp = 2.0f * (sqrtf( d_v[index].e * d_v[index].o) * cosf(h1 * M_PIf) + sqrtf(d_v[index].s * d_v[index].n) * cosf(h2 * M_PIf));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_w(Node* d_v, int i, int j, int tam, double h1, double h2) {
	double resp = 2.0f / (1.0f + sqrtf(1.0f - powf(f_q(d_v, i, j, tam, h1, h2), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ double f_v(Node* d_v, int i, int j, int tam) {
	double resp = d_v[i * tam + j].temp;
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}

// interação de gauss_seidel em uas etapas
__global__ void kernel_vermelho(Node* d_v, double h1, double h2, double n1, double n2, int tam) {
	int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int i = 1 + blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n2 + 1 && j < n1 + 1 && (i+j) % 2 == 0) {
		Node* ptr = &d_v[i * tam + j];
		double w = ptr->w;
		ptr->temp = (1.0f - w) * f_v(d_v, i, j, tam) + w *
						(ptr->o * f_v(d_v, i - 1, j, tam) +
						 ptr->e * f_v(d_v, i + 1, j, tam) +
						 ptr->s * f_v(d_v, i, j - 1, tam) +
						 ptr->n * f_v(d_v, i, j + 1, tam));
	}
}
__global__ void kernel_azul(Node* d_v, double h1, double h2, double n1, double n2, int tam) {
	int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int i = 1 + blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n2 + 1 && j < n1 + 1 && (i+j) % 2 == 1) {
		Node* ptr = &d_v[i * tam + j];
		double w = ptr->w;
		ptr->temp = (1.0f - w) * f_v(d_v, i, j, tam) + w *
					   (ptr->o * f_v(d_v, i - 1, j, tam) +
					    ptr->e * f_v(d_v, i + 1, j, tam) +
					    ptr->s * f_v(d_v, i, j - 1, tam) +
					    ptr->n * f_v(d_v, i, j + 1, tam));
	}
}

// funções para inicialização!
__global__ void kernel_set_borda(Node* d_v,double un,double us,double ue,double uo,int tam, int n1, int n2){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < n2+2){
		d_v[i*tam+0].temp = uo; 	 // borda esquerda
		d_v[i*tam+n1+1].temp = ue;   // borda direita
	}
	if(i < n1+2){
		d_v[0*tam+i].temp = un;		 // borda da superior
		d_v[(n2+1)*tam+i].temp = us; // borda da inferior
	}
}

__global__ void	kernel_set_A_B (double* d_a,double* d_b,int n1,int n2,double h1,double h2,int tam){
	int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int i = 1 + blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n2 + 1 && j < n1 + 1) {
		double x = i * h1;
		double y = j * h2;

		d_a[i * tam + j] = FATORf * x * (1.0f - x) * (0.5f - y);
		d_b[i * tam + j] = FATORf * y * (1.0f - y) * (x - 0.5f);
	}
}

__global__ void	kernel_set_V (Node* d_v,double* d_a,double* d_b,int n1,int n2,double h1,double h2,int tam,double denominadro2,double denominadro1, double temp_media){
	int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int i = 1 + blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n2 + 1 && j < n1 + 1) {
		Node* ptr = &d_v[i * tam + j];
		ptr->temp = temp_media;
		ptr->n = f_n(d_b, i, j, tam, h2, denominadro2);
		ptr->s = f_s(d_b, i, j, tam, h2, denominadro2);
		ptr->e = f_e(d_a, i, j, tam, h1, denominadro1);
		ptr->o = f_o(d_a, i, j, tam, h1, denominadro1);
	}
}

__global__ void kernel_set_W (Node*d_v, int n1, int n2, int tam, double h1, double h2){
	int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int i = 1 + blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n2 + 1 && j < n1 + 1) {
		d_v[i * tam + j].w = f_w(d_v, i, j, tam, h1, h2);
	}
}

// inicialização das variáveis globais e da matriz
void init_gpu() {
	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);
	denominadro1 = (4 * (1 + (h1 * h1 / (h2 * h2))));
	denominadro2 = (4 * (1 + (h2 * h2 / (h1 * h1))));

	int tam = n1 + 2;
	double temp_media = (us+un+ue+uo) / 4.0;

	cudaEvent_t start, stop;

	v = (Node*) malloc((n1 + 2) * (n2 + 2) * sizeof(Node));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_v, (n1 + 2) * (n2 + 2) * sizeof(Node)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_a, (n1 + 2) * (n2 + 2) * sizeof(double)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_b, (n1 + 2) * (n2 + 2) * sizeof(double)));

	if (v == NULL || d_v == NULL || d_a == NULL || d_b == NULL) {
		printf("---Erro de malloc");
		exit(EXIT_FAILURE);
	}

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start));

	// inicializando contorno
	int nBlocos = (n1 > n2) ? (n1+2)/MAX_THREAD + 1 : (n2+2)/MAX_THREAD + 1;

	if(verboso) printf(" --- Set_bordas\n");;
	kernel_set_borda <<< nBlocos , MAX_THREAD >>> (d_v, un, us, ue, uo, tam, n1, n2);
	CUDA_SAFE_CALL(cudaGetLastError());

	// incializando a e b
	N_threads_x = THREADS;
	N_threads_y = THREADS;
	N_blocs_x = (n2 + 2) / THREADS + 1;
	N_blocs_y = (n1 + 2) / THREADS + 1;

	dim3 threads_per_blocos (N_threads_x, N_threads_y);
	dim3 grade_blocos (N_blocs_x , N_blocs_y);

	if(verboso) printf(" --- Set_A_B\n");
	kernel_set_A_B <<< grade_blocos , threads_per_blocos >>> (d_a, d_b, n1, n2, h1, h2, tam);
	CUDA_SAFE_CALL(cudaGetLastError());


	if(verboso) printf(" --- Set_temps_V\n");
	kernel_set_V <<< grade_blocos , threads_per_blocos >>> (d_v, d_a, d_b, n1, n2, h1, h2, tam, denominadro2, denominadro1, temp_media);
	CUDA_SAFE_CALL(cudaGetLastError());

	if(verboso) printf(" --- Set_W\n");
	kernel_set_W <<< grade_blocos , threads_per_blocos >>> (d_v, n1, n2, tam, h1, h2);
	CUDA_SAFE_CALL(cudaGetLastError());

	CUDA_SAFE_CALL(cudaEventRecord(stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	// CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop)); // calculando tempo de init
}

// funcão que trata tudo que for necessario para executar em gpu
void gauss_seidel_gpu() {
	int tam = n1 + 2;
	cudaEvent_t start, stop;

	N_blocs_x = ((n1 + 2) / N_threads_x) + 1;
	N_blocs_y = ((n2 + 2) / N_threads_y) + 1;

	dim3 threads_per_blocos(N_threads_x, N_threads_y);
	dim3 grade_blocos(N_blocs_x , N_blocs_y);

	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start));

	int k;
	for ( k = 0; k < iteracoes; ++k) {

		kernel_vermelho <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam);

		kernel_azul <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam);
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&delta_eventos, start, stop));

	GET_TIME(inicio);
	CUDA_SAFE_CALL(cudaMemcpy( v, d_v, (n1 + 2) * (n2 + 2) * sizeof(Node), cudaMemcpyDeviceToHost));
	GET_TIME(fim);
	tempo_volta = fim - inicio;
}

// para o sequencial
void init() {
	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);
	denominadro1 = (4.0 * (1.0 + (h1 * h1 / (h2 * h2))));
	denominadro2 = (4.0 * (1.0 + (h2 * h2 / (h1 * h1))));

	v = (Node*) malloc((n1 + 2) * (n2 + 2) * sizeof(Node));
	a = (double*) malloc((n1 + 2) * (n2 + 2) * sizeof(double));
	b = (double*) malloc((n1 + 2) * (n2 + 2) * sizeof(double));

	if (v == NULL || a == NULL || b == NULL) {
		printf("---Erro de malloc");
		exit(EXIT_FAILURE);
	}

	double x, y;
	int i, j;

	int tam = n1 + 2;
	int temp_media = (ue+uo+un+us)/4.0 ;

	Node* pt;

  	// casos de borda
	for ( i = 0; i < n2 + 2; i++ ) {
		v[i * tam + 0].temp = uo;
		v[i * tam + n1 + 1].temp = ue;
	}
	for ( j = 0; j < n1 + 2; j++ ) {
		v[0 * tam + j].temp = un;
		v[(n2 + 1) * tam + j].temp = us;
	}

	// inicializando a e b
	for ( i = 0; i < n2 + 2; i++ ) {
		for ( j = 0; j < n1 + 2; j++ ) {
			x = i * h1;
			y = j * h2;
			a[i * tam + j] = FATOR * x * (1.0 - x) * (-y + 0.5);
			b[i * tam + j] = FATOR * y * (1.0 - y) * (x - 0.5);
		}
	}

	// inicializando as estruturas
	for ( i = 1; i < n2 + 1; i++ ) {
		for ( j = 1; j < n1 + 1; j++ ) {
			pt = &v[i * tam + j];
			pt->temp = temp_media;
			pt->n = f_n(i,j,tam);
			pt->s = f_s(i,j,tam);
			pt->e = f_e(i,j,tam);
			pt->o = f_o(i,j,tam);
		}
	}

	// calculando w
	for ( i = 1; i < n2 + 1; i++ ) {
		for ( j = 1; j < n1 + 1; j++ ) {
			v[i * tam + j].w = f_w(i,j,tam);
		}
	}

	free(a);
	free(b);
}

void gauss_seidel_sequencial() {
	int tam = n1+2;
	int i, j, k;
	double aux;
	Node* pt;

	for ( k = 0; k < iteracoes; ++k) {
		for ( i = 1; i < n2 + 1; i += 1) {
			for ( j = i % 2 ? 1 : 2 ; j < n1 + 1; j += 2) {
				pt = &v[i * tam + j];
				if (verboso) printf("verm\t[%d][%d]\n", i, j);
				if (verboso) printf("\tw:%f\n", pt->w);
				aux =
					pt->o * f_v(i- 1, j, tam) +
					pt->e * f_v(i + 1, j, tam) +
					pt->s * f_v(i, j - 1, tam) +
					pt->n * f_v(i, j + 1, tam);
				if (verboso) printf("\taux:%f\n", aux);
				pt->temp = (1 - pt->w) * pt->temp + pt->w * aux;
				if (verboso) printf("\td_v:%f\n", pt->temp);
			}
		}
		for ( i = 1; i < n2 + 1; i += 1) {
			for ( j = i % 2 ? 2 : 1; j < n1 + 1; j += 2) {
				if (verboso) printf("azul\t[%d][%d]\n", i, j);
				pt = &v[i * tam + j];
				if (verboso) printf("\tw:%f\n", pt->w);
				aux =
					pt->o * f_v(i- 1, j, tam) +
					pt->e * f_v(i + 1, j, tam) +
					pt->s * f_v(i, j - 1, tam) +
					pt->n * f_v(i, j + 1, tam);
				if (verboso) printf("\taux:%f\n", aux);
				pt->temp = (1 - pt->w) * pt->temp + pt->w * aux;
				if (verboso) printf("\td_v:%f\n", pt->temp);
			}
		}
	}
}

// checa se a entrada da main está correta e incicializa as falgs
int checa_entrada(const int argc, const char** argv){
	if( argc < 4){
		printf("Falta de parametros!!!\nEntre com %s <numero de pontos em x> <numero de pontos em y> <numero de interações>\n", argv[0]);
		return -1;
	}
	if (argv[1][0] == '-' || argv[2][0] == '-' || argv[3][0] == '-'){
		printf("Falta de parametros!!!\nEntre com %s <numero de pontos em x> <numero de pontos em y> <numero de interações>\n", argv[0]);
		return -1;
	}
	if( argc > 4){
		for (int i = 4; i < argc; ++i){ // começa do 4 pos se os 3 primeiros parametros devem ser obrigatorios e já analizados!
			if(argv[i][0] == '-'){

				switch (argv[i][1]){
					case 'v':
					case 'V':
					verboso = 1;
					break;

					case 'g':
					case 'G':
					executa_cpu = 0;;
					break;

					case 'c':
					case 'C':
					executa_gpu = 0;;
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

		GET_TIME(inicio);
		init_gpu();
		GET_TIME(fim);

		time_init_gpu = fim - inicio;

		if(verboso) printf(" Executando\n");

		// GET_TIME(inicio);
		gauss_seidel_gpu();
		// GET_TIME(fim);

		// tempo_gpu = fim - inicio;

		// impressao em arquivo texto
		file = fopen("out_gpu_tabelas.txt", "w");
		for ( int i = 0; i < n2 + 2; i++ ) {
			for ( int j = 0; j < n1 + 2; j++ ) {
				fprintf(file, "%6.3f ", v[i * (n1 + 2) + j].temp );
			}
			fprintf(file, "\n");
		}
		if(verboso) printf("--Acabou\n");

		fclose(file);
		free(v);

		if(verboso)PLOT(v,temp)
	}

	/*-------------- CPU -----------------*/
	if( executa_cpu){

		if(verboso) printf("Iniciando CPU...\n");

		GET_TIME(inicio);
		init();
		GET_TIME(fim);

		time_init_seq = fim - inicio;

		if(verboso) printf(" Executando\n");

		GET_TIME(inicio);
		gauss_seidel_sequencial();
		GET_TIME(fim);

		tempo_seq = fim - inicio;

		// impressao em arquivo texto
		file = fopen("out_cpu_tabelas.txt", "w");
		for ( int i = 0; i < n2 + 2; i++ ) {
			for ( int j = 0; j < n1 + 2; j++ ) {
				fprintf(file, "%6.3f ", v[i * (n1 + 2) + j].temp );
			}
			fprintf(file, "\n");
		}
		fclose(file);
		free(v);

		if(verboso) PLOT(v,temp)
	}

	//------------------------------- imprime dos tempos de execucao ----------------------//

	printf("Threads: %d\n", THREADS);
	printf("Pontos: %d x %d\n", n1, n2);
	printf("Iteracões: %d\n", iteracoes);
	printf("Tempo total sequencial          = %f seg \n", tempo_seq + time_init_seq);
	printf("Tempo total GPU                 = %f seg \n\n", delta_eventos / 1000 + time_init_gpu + tempo_volta);
	printf("Tempo trabalho sequencial       = %f seg \n", tempo_seq);
	printf("Tempo trabalho paralelo         = %f seg \n\n", delta_eventos / 1000);
	printf("Tempo inicialização sequencial  = %f seg \n", time_init_seq);
	printf("Tempo inicialização por gpu 	= %f seg \n", time_init_gpu);
	printf("Tempo volta da gpu 	= %f seg \n", tempo_volta);

	CUDA_SAFE_CALL(cudaDeviceReset());

	return 0;
}
