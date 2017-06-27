#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 8
#define M 2560
#define K 2560 
#define N 2560 

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *file, const int line )
{
	if( CUDA_SUCCESS != err) {
		fprintf(stderr,
				"CUDA Driver API error = %04d from file <%s>, line %i.\n",
				err, file, line );
		exit(-1);
	}   
}

/*
typedef struct {
	int height;
	int width;
	float *elements;
} Matrix;
*/

struct Matrix {
	int height;
	int width;
	float *elements;
};

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
	float cValue = 0.; 
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;
	for (int i = 0; i < A.width; ++i) {
		cValue += A.elements[row * A.width + i] * B.elements[i * B.width + col];
	}
	C.elements[row * C.width + col] = cValue;
}

void MatMulGPU(const Matrix A, const Matrix B, Matrix C) {
	Matrix d_A, d_B, d_C;
	d_A.width = A.width; d_A.height = A.height;
	d_B.width = B.width; d_B.height = B.height;
	d_C.width = C.width; d_C.height = C.height;
	size_t size_A = A.width * A.height * sizeof(float);
	size_t size_B = B.width * B.height * sizeof(float);
	size_t size_C = C.width * C.height * sizeof(float);
	checkCudaErrors(cudaMalloc(&d_A.elements, size_A));
	checkCudaErrors(cudaMalloc(&d_B.elements, size_B));
	checkCudaErrors(cudaMalloc(&d_C.elements, size_C));
	checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B.elements, B.elements, size_B, cudaMemcpyHostToDevice));

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(A.height / dimBlock.x, B.width / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	checkCudaErrors(cudaMemcpy(C.elements, d_C.elements, size_C, cudaMemcpyDeviceToHost));

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
	//cudaDeviceSynchronize();
}

void MatMulCPU(const Matrix A, const Matrix B, Matrix C) {
	for (int i = 0; i < C.height; ++i) {
		for (int j = 0; j < C.width; ++j) {
			C.elements[i * C.width + j] = 0.;
			for (int k = 0; k < A.width; ++k) {
				C.elements[i * C.width + j] += A.elements[i * A.width + k] * B.elements[k * B.width + j];
			}
		}
	}
}

int main(void)
{
	// init Matrix A, B
	Matrix h_A, h_B, h_C;
	h_A.height = M; h_A.width = K;
	h_A.elements = (float *) malloc(M * K * sizeof(float));
	for (int i = 0; i < M * K; ++i) h_A.elements[i] = 0.1;

	h_B.height = K; h_B.width = N;
	h_B.elements = (float *) malloc(K * N * sizeof(float));
	for (int i = 0; i < K * N; ++i) h_B.elements[i] = 0.2;

	h_C.height = M; h_C.width = N;
	h_C.elements = (float *) malloc(M * N * sizeof(float));

	MatMulGPU(h_A, h_B, h_C);
	//MatMulCPU(h_A, h_B, h_C);

/*
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%f ", h_C.elements[j + i * M]);
		}
		printf("\n");
	}
*/

	free(h_A.elements);
	free(h_B.elements);
	free(h_C.elements);
	return 0;
}
