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
	int width;
	int height;
	int stride;
	float *elements;
} Matrix;
*/

struct Matrix {
	int width;
	int height;
	int stride;
	float *elements;
};

__device__ float GetElement(const Matrix A, int row, int col) {
	return A.elements[row * A.stride + col];
} 

__device__ void SetElement(Matrix A, int row, int col, float value) {
	A.elements[row * A.stride + col] = value;
} 

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
	int blockRow = blockIdx.x;
	int blockCol = blockIdx.y;

	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	float cValue = 0.;
	int row = threadIdx.x;
	int col = threadIdx.y;

	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		__syncthreads();
		for (int e = 0; e < BLOCK_SIZE; ++e) {
			cValue += As[row][e] * Bs[e][col];
		}
		__syncthreads();
	}
	SetElement(Csub, row, col, cValue);
}

void MatMulGPU(const Matrix A, const Matrix B, Matrix C) {
	Matrix d_A, d_B, d_C;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
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
