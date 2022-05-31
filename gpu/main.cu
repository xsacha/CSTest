#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda.h>

#include "cudaHelper.h"

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



    // Perform explicit
    {
        printf("Explicit, single-precision\n");
    }

    // Perform implicit
	{
		printf("Implicit, single-precision, discretise S in to 514 slices, timestamps = 200\n");
        // Excludes end points Smin=0.0 and Smax=50.0
		num_dissteps = 512;
    }

    // Perform Crank Nicolson
	{
		printf("CrankNicolson, double-precision, discretise S in to 2050 slices, timesteps = 200\n");
        // Excludes end points Smin=0.0, Smax=50.0
        num_dissteps = 2048;
    }

    return 0;
}