#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda.h>

#include "cudaHelper.h"
#include "formula.h"
#include "finiteDiff.h"

int main()
{
	// European calls to price, multiples of block size
	const int num_options = 1024;
	int num_dissteps;
	int num_blocks;
	int num_threads;

	// Record time via CUDA API
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Set test parameters to match reference
	double S0 = 7.7;
	double sigma = 0.3;
	double r = 0.04;
	double T = 16.0 / 365;
	double K = 7.51;
	double Smax = 50.0f;

	constexpr unsigned TIMESTEPS = 200;

    // Use Formula (not parallel)
    {
        double* resultGPU;
        checkCudaErrors(cudaMalloc((void**)&resultGPU, sizeof(double)));
        cudaEventRecord(start);
        UseFormula<<<1,1>>>(resultGPU, T, S0, r, K, sigma);
        double result;
        checkCudaErrors(cudaMemcpy(&result, resultGPU, sizeof(double), cudaMemcpyDeviceToHost));

        cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float millisecondsElapsed;
		cudaEventElapsedTime(&millisecondsElapsed, start, stop);

        printf("Using Black-Scholes formula, result is: %lf\n", result);
        printf("Consumed %f ms\n", millisecondsElapsed);
    }

    // Perform explicit time-marching
    {
        printf("Explicit, single-precision\n");
        num_dissteps = 1024;
        num_blocks = 1;
        num_threads = TIMESTEPS - 1;

        // Parallel only by requiring multiple calculations
        int memsize = num_dissteps * sizeof(double);
		double* d_x;
		checkCudaErrors(cudaMalloc((void**)&d_x, memsize));

        cudaEventRecord(start);
        ExplicitMethod<<<num_blocks, num_threads>>>(sigma, Smax, K, T, r, d_x);
        double* x = (double*)malloc(memsize);
		checkCudaErrors(cudaMemcpy(x, d_x, memsize, cudaMemcpyDeviceToHost));

        cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float millisecondsElapsed;
		cudaEventElapsedTime(&millisecondsElapsed, start, stop);

        // Get price by interpolation method
        double ds = Smax / (num_dissteps + 1);
        printf("Using Explicit method when spot price is %lf: %lf\n", S0, getPrice(S0, ds, x));
        printf("Consumed %f ms\n", millisecondsElapsed);
    }

    // Perform implicit
	{
		printf("Implicit, single-precision, discretise S in to 514 slices, timestamps = 200\n");
        // Excludes end points Smin=0.0 and Smax=50.0
		num_dissteps = 512;

        // Compute the corresponding upper boundary. (we will treat Smin=0.0 in this case)
		float* upperbound;
		float* d_upperbound;
		upperbound = (float*)malloc((TIMESTEPS + 1) * sizeof(float));
		checkCudaErrors(cudaMalloc((void**)&d_upperbound, (TIMESTEPS + 1) * sizeof(float)));
		for (int i = 0; i <= TIMESTEPS; i++)
		{
            //vanilla European call option
			upperbound[i] = (float)((Smax - K) * exp(-r * T / TIMESTEPS * i));
		}
		checkCudaErrors(cudaMemcpy(d_upperbound, upperbound, (TIMESTEPS + 1) * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Perform Crank Nicolson
	{
		printf("CrankNicolson, double-precision, discretise S in to 2050 slices, timesteps = 200\n");
        // Excludes end points Smin=0.0, Smax=50.0
        num_dissteps = 2048;
    }

    return 0;
}