/*
	Como todos os testes o acesso a memória por gpu foi pior que o sequencial!
	Voltams para a verção em que a gpu faz conta.
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
#define THREAD_X 16
#define THREAD_Y 16

#define M_PIf 3.141592653589f
#define FATOR 5.0
#define FATORf 5.0f

#define PLOT(m,k) {int _i, _j; \
	for ( _i = 1; _i < n2 + 1; ++_i){	\
		for ( _j = 1; _j < n1 + 1; ++_j){\
			printf("%f ", m[_i*(n2+2) + _j].k);\
		}	\
		printf("\n");\
	}\
	printf("\n");\
}
#define PLOT2(m){int _i, _j; \
	for ( _i = 1; _i < n2 + 1; ++_i){	\
		for ( _j = 1; _j < n1 + 1; ++_j){\
			printf("%f ", m[_i*(n2+2) + _j]);\
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
	float temp;
	float n;
	float s;
	float e;
	float o;
	float w;
} Node;

// flags para o programa! 
int verboso = 0;
int executa_gpu = 1;
int executa_cpu = 1;

int interacoes;

int N_threads_x = THREAD_X;
int N_threads_y = THREAD_Y;
int N_blocs_x = 1;
int N_blocs_y = 1;

/* faltam as variavas de tempo*/

// matrizes cpu da chapa
Node *v;
float *retorno_gpu;
float *a;
float *b;

// matrizes gpu da chapa
float *d_v;
float *d_a;
float *d_b;

float h1 = 0; // distancia orizontal dos potos
float h2 = 0; // distancia vertical dos potos
float denominadro1; // = (4 * (1 + (h1 * h1 / (h2 * h2))))
float denominadro2; // = (4 * (1 + (h2 * h2 / (h1 * h1))))
int n1 = 1;   // quantidade de potnos em uma linha orizontal
int n2 = 1;   // quantidade de potnos em uma linha vertical

float uo = 0;  // temperatura fixa a oeste
float ue = 10; // temperatura fixa a este
float us = 5;  // temperatura fixa a sul
float un = 5;  // temperatura fixa a norte

// variaveis para contar tempo
double inicio, fim;
double tempo_seq, tempo_gpu;
double time_init_seq, time_init_gpu;
double ida_gpu, volta_gpu;


// ------------------------- funções de contas cpu -------------------------- //
float f_a(int i, int j, int tam) {
	float resp = a[i * tam + j];

	return resp;
}
float f_b(int i, int j, int tam) {
	float resp = b[i * tam + j];

	return resp;
}
float f_o(int i, int j, int tam) {
	float resp = ( 2.0 + h1 * f_a(i, j, tam) ) / denominadro1;
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_e(int i, int j, int tam) {
	float resp = ( 2.0 - h1 * f_a(i, j, tam) ) / denominadro1;
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_s(int i, int j, int tam) {
	float resp = ( 2.0 + h2 * f_b(i, j, tam) ) / denominadro2;
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_n(int i, int j, int tam) {
	float resp = ( 2.0 - h2 * f_b(i, j, tam) ) / denominadro2;
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}

float f_q(int i, int j, int tam) {
	int index = i*tam +j;
	float resp = 2.0 * (sqrt( v[index].e * v[index].o) * cos(h1 * M_PI) + sqrt(v[index].s * v[index].n) * cos(h2 * M_PI));
	// printf("f_q(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_w(int i, int j, int tam) {
	float resp = 2.0 / (1.0 + sqrt(1.0 - pow(f_q(i, j, tam), 2)));
	// printf("f_w(%d,%d) = %f\n", i, j, resp);
	return resp;
}
float f_v(int i, int j, int tam) {
	float resp = v[i * tam + j].temp;
	// printf("f_v(%d,%d) = %f\n", i, j, resp);
	return resp;
}

// -------------------------- funções de contas gpu -------------------------- //
__device__ float f_a(int i, int j, float h1, float h2) {
	float x = i * h1;
	float y = j * h2;
	float resp = 500.0f * x * (1 - x) * (-y + 0.5f);
	return resp;
}
__device__ float f_b(int i, int j, float h1, float h2) {
	float x = i * h1;
	float y = j * h2;
	float resp = 500.0f * y * (1 - y) * (x - 0.5f);
	return resp;
}
__device__ float f_o(int i, int j, float h1, float h2) {
	float resp = ( 2 + h1 * f_a(i, j, h1, h2) ) / (4.0f * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_o(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_e(int i, int j, float h1, float h2) {
	float resp = ( 2 - h1 * f_a(i, j, h1, h2) ) / (4.0f * (1 + (h1 * h1 / (h2 * h2))));
	// printf("f_e(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_s(int i, int j, float h1, float h2) {
	float resp = ( 2 + h2 * f_b(i, j, h1, h2) ) / (4.0f * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_s(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_n(int i, int j, float h1, float h2) {
	float resp = ( 2 - h2 * f_b(i, j, h1, h2) ) / (4.0f * (1 + (h2 * h2 / (h1 * h1))));
	// printf("f_n(%d,%d) = %f\n", i, j, resp);
	return resp;
}
__device__ float f_q(int i, int j, float h1, float h2) {
	float resp = 2.0f * (sqrtf(f_e(i, j, h1, h2) * f_o(i, j, h1, h2)) * cosf(h1 * M_PIf) + sqrtf(f_s(i, j, h1, h2) * f_n(i, j, h1, h2)) * cosf(h2 * M_PIf));
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
	int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int i = 1 + blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n2 + 1 && j < n1 + 1 && (i+j)%2 == 0) { // quando i +j é par
		float w = f_w(i, j, h1, h2);
		float aux =
			f_o(i, j, h1, h2) * f_v(d_v, i, j-1, n1) +
			f_e(i, j, h1, h2) * f_v(d_v, i, j+1, n1) +
			f_s(i, j, h1, h2) * f_v(d_v, i-1, j, n1) +
			f_n(i, j, h1, h2) * f_v(d_v, i+1, j, n1);
		d_v[i * tam + j] = (1 - w) * f_v(d_v, i, j, n1) + w * aux;
	}
}
__global__ void kernel_azul(float* d_v, float h1, float h2, float n1, float n2, int tam) {
	int j = 1 + blockIdx.x * blockDim.x + threadIdx.x;
	int i = 1 + blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n2 + 1 && j < n1 + 1 && (i+j)%2 == 1) { // quando i +j é impar
		float w = f_w(i, j, h1, h2);
		float aux =
			f_o(i, j, h1, h2) * f_v(d_v, i, j-1, n1) +
			f_e(i, j, h1, h2) * f_v(d_v, i, j+1, n1) +
			f_s(i, j, h1, h2) * f_v(d_v, i-1, j, n1) +
			f_n(i, j, h1, h2) * f_v(d_v, i+1, j, n1);
		d_v[i * tam + j] = (1 - w) * f_v(d_v, i, j, n1) + w * aux;
	}
}



// inicialização das variáveis globais e da matriz
void init_gpu() {

	retorno_gpu = (float*) malloc((n1 + 2) * (n2 + 2) * sizeof(float));

	int i, j;
	int tam = n1 + 2;
	int temp_media = (ue+uo+un+us)/4.0 ;
	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);
  	
  	// casos de borda
	for ( i = 0; i < n2 + 2; i++ ) {
		retorno_gpu[i * tam + 0] = uo;
		retorno_gpu[i * tam + n1 + 1] = ue;
	}
	for ( j = 0; j < n1 + 2; j++ ) {
		retorno_gpu[0 * tam + j] = un;
		retorno_gpu[(n2 + 1) * tam + j] = us;
	}

	// Valor inicial
	for ( i = 1; i < n2 + 1; i++ ) {
		for ( j = 1; j < n1 + 1; j++ ) {
			retorno_gpu[i * tam + j] = temp_media;
		} 
	}

	// Ida para GPU
	GET_TIME(inicio);

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_v, (n1 + 2) * (n2 + 2) * sizeof(float)));
	if ( d_v == NULL) {
		printf("---Erro de malloc");
		exit(EXIT_FAILURE);
	}
	CUDA_SAFE_CALL(cudaMemcpy(d_v, retorno_gpu, (n1 + 2) * (n2 + 2) * sizeof(float), cudaMemcpyHostToDevice));

	GET_TIME(fim);

	ida_gpu = fim - inicio;
}

// funcão que trata tudo que for necessario para executar em gpu
void gauss_seidel_sequencial_gpu() {
	
	GET_TIME(inicio);
	
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
	for ( k = 0; k < interacoes; ++k) {

		kernel_vemelho <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam);

		kernel_azul <<< grade_blocos , threads_per_blocos >>> (d_v, h1, h2, n1, n2, tam);
	}

	CUDA_SAFE_CALL(cudaEventRecord(stop));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	GET_TIME(fim);
	
	tempo_gpu = fim - inicio;

	// Volta da GPU
	GET_TIME(inicio);
	CUDA_SAFE_CALL(cudaMemcpy( retorno_gpu, d_v, (n1 + 2) * (n2 + 2) * sizeof(float), cudaMemcpyDeviceToHost));
	GET_TIME(fim);

	volta_gpu = fim - inicio;
}

// para o sequencial
void init() {
	GET_TIME(inicio);
	
	h1 = 1.0 / (n1 + 1);
	h2 = 1.0 / (n2 + 1);
	denominadro1 = (4.0 * (1.0 + (h1 * h1 / (h2 * h2))));
	denominadro2 = (4.0 * (1.0 + (h2 * h2 / (h1 * h1))));

	v = (Node*) malloc((n1 + 2) * (n2 + 2) * sizeof(Node));
	a = (float*) malloc((n1 + 2) * (n2 + 2) * sizeof(float));
	b = (float*) malloc((n1 + 2) * (n2 + 2) * sizeof(float));

	if (v == NULL || a == NULL || b == NULL) {
		printf("---Erro de malloc");
		exit(EXIT_FAILURE);
	}

	float x, y;
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
			x = j * h1;
			y = i * h2;
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
	
	GET_TIME(fim);
	time_init_seq = fim - inicio;
	
	free(a);
	free(b);
}

void gauss_seidel_sequencial() {
	int tam = n1+2;
	int i, j, k;
	float aux;
	Node* pt;

	for ( k = 0; k < interacoes; ++k) {
		for ( i = 1; i < n2 + 1; i += 1) {
			for ( j = i % 2 ? 1 : 2 ; j < n1 + 1; j += 2) {
				pt = &v[i * tam + j];

				aux = 
				pt->o * f_v(i  , j-1 ,tam) +
				pt->e * f_v(i  , j+1 ,tam) +
				pt->s * f_v(i-1, j   ,tam) +
				pt->n * f_v(i+1, j   ,tam);
				pt->temp = (1 - pt->w) * pt->temp + pt->w * aux;
			}
		}
		for ( i = 1; i < n2 + 1; i += 1) {
			for ( j = i % 2 ? 2 : 1; j < n1 + 1; j += 2) {
				pt = &v[i * tam + j];
				
				aux = 
				pt->o * f_v(i  , j-1 ,tam) +
				pt->e * f_v(i  , j+1 ,tam) +
				pt->s * f_v(i-1, j   ,tam) +
				pt->n * f_v(i+1, j   ,tam);
				pt->temp = (1 - pt->w) * pt->temp + pt->w * aux;
			}
		}
	}
}

// checa se a entrada da main está correta e incicializa as falgs
int checa_entrada(const int argc, const char** argv){
	if( argc < 4){
		printf("Falta de parametros!!! Entre com %s <numero de pontos em x> <numero de pontos em y> <numero de interações>\n", argv[0]);
		return -1;
	}
	if (argv[1][0] == '-' || argv[2][0] == '-' || argv[3][0] == '-'){
		printf("Falta de parametros!!! Entre com %s <numero de pontos em x> <numero de pontos em y> <numero de interações>\n", argv[0]);
		return -1;
	}
	if( argc > 4){
		for (int i = 4; i < argc; ++i){ // começa do 4 pos se os 3 primeiros parametros devem ser obrigatorios e já analizados!
			if(argv[i][0] == '-'){

				switch (argv[i][1]){
					case 'v':
					case 'V':
					verboso = 1; // executa me modo verboso para companhar log de execução
					break;

					case 'g':
					case 'G':
					executa_cpu = 0; // Só executa GPU
					break;
					
					case 'c':
					case 'C':
					executa_gpu = 0; // Só executa CPU
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
	n1 = atoi(argv[1]); // argumentos garantidos por conta da função checa_entrada()
	n2 = atoi(argv[2]);
	interacoes = atoi(argv[3]);

	/*-------------- GPU -----------------*/
	if(executa_gpu){ // flag de execulçao setada pela função checa_entrada()
		
		if(verboso) printf("Iniciando GPU...\n");
		init_gpu();

		if(verboso) printf(" Executando\n");
		gauss_seidel_sequencial_gpu();		

		// impressao em arquivo texto
		file = fopen("out_gpu.txt", "w"); 
		for ( int i = 0; i < n2 + 2; i++ ) {
			for ( int j = 0; j < n1 + 2; j++ ) {
				fprintf(file, "%6.3f,", retorno_gpu[i * (n1 + 2) + j] );
			}
			fprintf(file, "\n");
		}
		if(verboso) printf("--Acabou\n");

		fclose(file);
		free(v);
		
		if(verboso)PLOT2(retorno_gpu)
	}

	/*-------------- CPU -----------------*/
	if( executa_cpu){ // flag de execulçao setada pela função checa_entrada()

		if(verboso) printf("Iniciando CPU...\n");
		init();

		if(verboso) printf(" Executando\n");
		GET_TIME(inicio);
		gauss_seidel_sequencial();
		GET_TIME(fim);

		tempo_seq = fim - inicio;

		// impressao em arquivo texto
		file = fopen("out_cpu.txt", "w");
		for ( int i = 0; i < n2 + 2; i++ ) {
			for ( int j = 0; j < n1 + 2; j++ ) {
				fprintf(file, "%6.3f,", v[i * (n1 + 2) + j].temp );
			}
			fprintf(file, "\n");
		}
		fclose(file);
		free(v);

		if(verboso) PLOT(v,temp)
	}

	//------------------------------- imprime dos tempos de execucao ----------------------//
	
	printf("Tempo total sequencial          = %f seg \n", tempo_seq + time_init_seq);
	printf("Tempo total GPU                 = %f seg \n\n", tempo_gpu + ida_gpu + volta_gpu);
	printf("Tempo trabalho sequencial       = %f seg \n", tempo_seq);
	printf("Tempo trabalho paralelo         = %f seg \n\n", tempo_gpu);
	printf("Tempo inicialização sequencial  = %f seg \n", time_init_seq);
	printf("Tempo de ida para gpu           = %f seg \n", ida_gpu);
	printf("Tempo de volta para gpu         = %f seg \n", volta_gpu);

	CUDA_SAFE_CALL(cudaDeviceReset());

	return 0;
}











