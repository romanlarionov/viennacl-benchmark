
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <benchmark/flops.h>
#include <fstream>

void FillDoubleComplex(cuDoubleComplex *mat, int size1, int size2)
{
    int i;
    for (i = 0; i < (double)size1 * (double)size2; i++)
    {
        mat[i].x = (double) rand() / (double) RAND_MAX;
        mat[i].y = (double) rand() / (double) RAND_MAX;
    }
}

int main(int argc, char **argv)
{
	std::ofstream stream;
    stream.open("cublas_results.txt");

    // initialize cublas
    cublasStatus_t init_status = cublasInit();
    if (init_status != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!!! Cublas initialization error. Exiting.\n");
        return EXIT_FAILURE;
    }

    cublasHandle_t handle;
    cublasStatus_t cublas_statis = cublasCreate(&handle);

    if (cublas_statis != CUBLAS_STATUS_SUCCESS) {
        printf("!!!!!! Cublas initialization error. Exiting.\n");
        return EXIT_FAILURE;
    }

    int num_runs = 50;
	double total_time = 0.0;
	double total_gflops = 0.0;

    // double complex matrix computation loop
    unsigned int loop = 0;
    for (loop = 1; loop <= num_runs; ++loop) {
        int M = 4000;
        int N = 4000;
        int K = 4000;

        cuDoubleComplex alpha_dc, beta_dc;

        alpha_dc.x = (double) rand() / (double) RAND_MAX;
        alpha_dc.y = (double) rand() / (double) RAND_MAX;
        beta_dc.x = (double) rand() / (double) RAND_MAX;
        beta_dc.y = (double) rand() / (double) RAND_MAX;

        cuDoubleComplex *host_A, *host_B, *host_C, *host_result;
        cuDoubleComplex *device_A, *device_B, *device_C;

        // allocate host memory
        host_A = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * M * K);
        host_B = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * K * N);
        host_C = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * M * N);

        // allocate device memory
        cudaMalloc((void **) &device_A, sizeof(cuDoubleComplex) * M * K);
        cudaMalloc((void **) &device_B, sizeof(cuDoubleComplex) * K * N);
        cudaMalloc((void **) &device_C, sizeof(cuDoubleComplex) * M * N);

        // fill matrices
        FillDoubleComplex(host_A, M, K);
        FillDoubleComplex(host_B, K, N);
        FillDoubleComplex(host_C, M, N);

		// timing code
		struct timeval tval;
		gettimeofday(&tval, NULL);
		double start_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);

        // transfer from host to device
        cudaMemcpy(device_A, host_A, M * K * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
            printf("host A to device. cudamemcpy error -- %s\n", cudaGetErrorString(error));

        cudaMemcpy(device_B, host_B, K * N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaError_t code96 = cudaGetLastError();
        if (code96 != cudaSuccess)
            printf("host B to device. error -- %s\n", cudaGetErrorString(code96));

        cudaMemcpy(device_C, host_C, M * K * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        cudaError_t code92 = cudaGetLastError();
        if (code92 != cudaSuccess)
            printf("host C to device. error -- %s\n", cudaGetErrorString(code92));

        // run parallel product
        cublasStatus_t res = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha_dc, device_A, M, device_B, K,
                                         &beta_dc, device_C, M);
        if (res != CUBLAS_STATUS_SUCCESS) {
            printf("Execution error! %i\n", (int) res);
        }

        // transfer result to host
        host_result = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * M * K);
        cudaMemcpy(host_result, device_C, M * N * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cudaError_t code93 = cudaGetLastError();
        if (code93 != cudaSuccess)
            printf("device C to host. error -- %s\n", cudaGetErrorString(code93));

		// end timer
		gettimeofday(&tval, NULL);
		double end_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);

        // output
        double sec = static_cast<double>(end_time - start_time) / 1000000.0;
        double GFLOPS = FLOPS_ZGEMM(M, N, K) / (1000000000.0);
		total_gflops += GFLOPS;
		total_time += sec;

		stream << "Run : " << loop << "\t\t";
		stream << "Real Execution Time: " << sec << "s\t";
		stream << "GFLOPs per second: " << (GFLOPS / sec) << "\n";

        // free mem
        free(host_A);
        free(host_B);
        free(host_C);
        free(host_result);
        cudaFree(device_A);
        cudaFree(device_B);
        cudaFree(device_C);
    }

    // performance marking
    double mean_time = total_time / num_runs;
    double mean_gflops_per_second = total_gflops / total_time;

    stream << "Number runs: " << num_runs << "\n" << "Mean Time " << mean_time << ".\n" <<
            "Mean GFLOPS per second: " << mean_gflops_per_second << ".\n" <<
			"Total Execution Time: " << total_time << "\n";

    stream.close();

    // deinitialize
    cublasDestroy(handle);
    cudaDeviceReset();
    cublasShutdown();
}


