#pragma once
#include "optimisations.h"

// Using REAL template to try to maintain double and float options

// Explicit time marching method
template <typename REAL>
__global__ void ExplicitMethod(REAL sigma, REAL Smax, REAL K, REAL T, REAL r, REAL* d_x, int imax, int jmax)
{
    // TODO: Maybe pass it in so no magic numbers?
	constexpr int steps = 1024;
	// Set of warps
	constexpr int subthreads = 32;
	// subsize = steps/subthreads (essentially the rows)
	constexpr int subsize = steps / subthreads;
    int t_id = threadIdx.x;
	int b_id = blockIdx.x;
    int curThread = t_id % subthreads;
    REAL dS = Smax / (steps + 1);
    // TODO: Also pass this in
	REAL dT = T / imax;

    __shared__ double V_[steps + 2];
	__shared__ double V_minus[steps + 2];
	V_[t_id + 1] = d_x[t_id + 1];
	__syncthreads();
    // Main time marching (not parallel)
    REAL S = (t_id + 1) * dS;
    for (int i = imax; i > 0; --i)
    {
        __syncthreads();
        // Payouts (0 for bottom)
        if (threadIdx.x == 0)
	    {
	    	V_minus[0] = 0.0;
            // steps is jmax for now
	    	V_minus[jmax] = ((Smax - K) * exp(-r * dT * i));
	    }
		// Pull it out instead of calc twice
		REAL halfdT = (REAL)0.5 * dT;
		REAL alpha = halfdT * sigma * sigma * S * S / (dS * dS);
		REAL beta = halfdT * r * S / dS;
		REAL a = -alpha + beta;
		REAL b = 1.0f + 2 * alpha + r * dT;
		REAL c = -alpha - beta;
        //REAL a = dt*(b(i,j)*0.5-a(i,j)/ds)/ds;
        //REAL b = 1.0-dt*c(i,j)+2.0*dt*a(i,j)/(ds*ds);
        //REAL c = -dt*(b(i,j)*0.5+a(i,j)/ds)/ds;
        // d term is 0
        V_minus[t_id + 1] = a * V_[t_id] + b * V_[t_id + 1] + c*V_[t_id + 2];
        __syncthreads();
        V_[t_id + 1] = V_minus[t_id + 1];
        __syncthreads();
    }

    d_x[t_id + 1] = V_[t_id + 1];
}

// Implicit Method of Black-Scholes. Using PCR+Thomas by shuffle to solve tri-diagonal system
// Try to maintain double and float options
template <typename REAL>
__global__ void ImplicitMethod(REAL sigma, REAL Smax, REAL K, REAL T, REAL r, REAL* d_x, REAL* d_upperbound)
{
	// TODO: Maybe pass it in so no magic numbers?
	constexpr int steps = 512;
	// Set of warps
	constexpr int subthreads = 32;
	// subsize = steps/subthreads (essentially the rows)
	constexpr int subsize = steps / subthreads;
	int t_id = threadIdx.x;
	int b_id = blockIdx.x;
	int curThread = t_id % subthreads;

	REAL dS = Smax / (steps + 1);
	// TODO: Also pass this in
	constexpr unsigned TIMESTEPS = 200;
	REAL dT = T / TIMESTEPS;

	REAL a[subsize];
	REAL b[subsize];
	REAL c[subsize];
	REAL d[subsize];
	REAL cmax;
	for (int i = 0; i < subsize; ++i)
	{
		// S from 1* dS to blockDim.x* dS (excluding Smin=0, Smax=(blockDim.x+1)* dS )
		REAL S = (curThread * subsize + i + 1) * dS;
		// Pull it out instead of calc twice
		REAL halfdT = (REAL)0.5 * dT;
		REAL alpha = halfdT * sigma * sigma * S * S / (dS * dS);
		REAL beta = halfdT * r * S / dS;
		a[i] = -alpha + beta;
		b[i] = 1.0f + 2 * alpha + r * dT;
		c[i] = -alpha - beta;
		// Store last row in to cmax
		if (i == subsize - 1)
		{
			cmax = c[i];
		}

		// The payout is the right hand side of the solution (tridiagonal system)
		// assuming vanilla European call option	
		d[i] = max(S - K, 0.0);
	}


	// Set edges to zero
	if (curThread == 0)
	{
		a[0] = 0.0;
	}
	if (curThread == subthreads - 1)
	{
		c[subsize - 1] = 0.0;
	}

	REAL ratio;
	REAL atem[subsize];
	REAL ctem[subsize];
	REAL dtem[subsize];
	// Part 1: Use modified Thomas algorithm to obtain equation system being expressed in terms of end values
	// a[i] * x[0] + x[i] + c[i] * x[SIZE-1] = d[i]
	for (int j = 0; j < TIMESTEPS; ++j)
	{
        // Transformation on the last row (upper bound)
		if (curThread == subthreads - 1)
		{
			d[subsize - 1] += -cmax * d_upperbound[j + 1];
		}

        // Forward pass
		for (int i = 0; i < 2; i++) {
			ratio = (REAL)1.0 / (b[i]);
			dtem[i] = ratio * d[i];
			atem[i] = ratio * a[i];
			ctem[i] = ratio * c[i];
		}

		for (int i = 2; i < subsize; i++) {
			ratio = (REAL)1.0 / (b[i] - a[i] * ctem[i - 1]);
			dtem[i] = ratio * (d[i] - a[i] * dtem[i - 1]);
			atem[i] = ratio * (-a[i] * atem[i - 1]);
			ctem[i] = ratio * c[i];
		}

		// Backward pass
		for (int i = subsize - 3; i > 0; i--)
		{
			dtem[i] = dtem[i] - ctem[i] * dtem[i + 1];
			atem[i] = atem[i] - ctem[i] * atem[i + 1];
			ctem[i] = -ctem[i] * ctem[i + 1];
		}

		ratio = (REAL)1.0 / ((REAL)1.0 - ctem[0] * atem[1]);
		dtem[0] = ratio * (dtem[0] - ctem[0] * dtem[1]);
		atem[0] = ratio * atem[0];
		ctem[0] = ratio * (-ctem[0] * ctem[1]);

		// Part 2: Use 'Cyclic Reduction' algorithm to get end values { d[0], d[SIZE-1]} of each thread
		PCR_2_warp(atem[0], ctem[0], dtem[0], atem[subsize - 1], ctem[subsize - 1], dtem[subsize - 1]);

		// Part 3: Solve all other central values by substituting end values in the equation system acquired in Part 1
		d[0] = dtem[0];
        d[subsize - 1] = dtem[subsize - 1];
		for (int i = 1; i < subsize - 1; i++)
		{
			dtem[i] = dtem[i] - atem[i] * dtem[0] - ctem[i] * dtem[subsize - 1];
			d[i] = dtem[i];
		}
	}

    // Send results back to our result vector
	for (int i = 0; i < subsize; ++i)
	{
		int index = (t_id + b_id * blockDim.x) * subsize + i;
		d_x[index] = d[i];
	}
}


// Crank Nicolson method of BS. CR + SPR solving tri-diagonal system
template <typename REAL>
__global__  void CrankNicolsonMethod(REAL sigma, REAL Smax, REAL K, REAL T, REAL r, REAL* d_a1, REAL* d_b1, REAL* d_c1, REAL* d_x, REAL* d_upperbound)
{
    int t_id = threadIdx.x;
	int b_id = blockIdx.x;

	// We set the odd (a1) and even (a2) coefficients separately as in the paper
	extern __shared__ char shared[];

	REAL* a2 = (REAL*)shared;
	// Index is odd (1, 3, 5, ..., 2 * blocksize - 1)
	REAL* b2 = (REAL*)&a2[blockDim.x + 1];
	REAL* c2 = (REAL*)&b2[blockDim.x + 1];
	REAL* d1 = (REAL*)&c2[blockDim.x + 1];
	REAL* d2 = (REAL*)&d1[blockDim.x + 1];

	// Remember: num_points = num_threads + 2 (end boundary points)
	REAL dS = Smax / (2 * blockDim.x + 1);
	// TODO: Probably pass it in
	constexpr unsigned TIMESTEPS = 200;
	REAL dT = T / TIMESTEPS;
	// S from [1 * dS] -> [blockDim.x * dS] (exclude boundary point)
	REAL S1 = (t_id * 2 + 1) * dS;
	REAL S2 = (t_id * 2 + 2) * dS;
	REAL alpha1 = 0.5 * dT * sigma * sigma * S1 * S1 / (dS * dS);
	REAL alpha2 = 0.5 * dT * sigma * sigma * S2 * S2 / (dS * dS);
	REAL beta1 = 0.5 * dT * r * S1 / dS;
	REAL beta2 = 0.5 * dT * r * S2 / dS;
	d_a1[t_id + b_id * blockDim.x] = -alpha1 + beta1;
	d_b1[t_id + b_id * blockDim.x] = 1.0f + 2 * alpha1 + r * dT;
	d_c1[t_id + b_id * blockDim.x] = -alpha1 - beta1;
	// Initialise the pay out as the right hand side of equation (tridiagonal system)
	// Vanilla European call option	
	d1[t_id] = max(S1 - K, 0.0);
	d2[t_id] = max(S2 - K, 0.0);


	if (t_id == 0)
	{
		d_a1[b_id * blockDim.x] = 0.0;
	}
	__syncthreads();



	// Main time-marching
    constexpr int ITERATION = 100;
	for (int k = 0; k < ITERATION; k++) {

		a2[t_id] = -alpha2 + beta2;
		b2[t_id] = 1.0f + 2 * alpha2 + r * dT;
		c2[t_id] = -alpha2 - beta2;
		if (t_id == blockDim.x - 1)
		{
			c2[t_id] = 0.0;
		}

		// Explicit and last row transformation 
		int up = 0;
        int down = 0;
		if (t_id > 0)
        {
            up = t_id - 1;
        }
		if (t_id < blockDim.x - 1)
        {
            down = t_id + 1;
        }
		__syncthreads();

		REAL d1_temp = -d_a1[t_id + b_id * blockDim.x] * d2[up] + (2 - d_b1[t_id + b_id * blockDim.x]) * d1[t_id] - d_c1[t_id + b_id * blockDim.x] * d2[t_id];
		REAL d2_temp = -a2[t_id] * d1[t_id] + (2 - b2[t_id]) * d2[t_id] - c2[t_id] * d1[down];
		__syncthreads();
		d1[t_id] = d1_temp; d2[t_id] = d2_temp;
		if (t_id == blockDim.x - 1) {
			REAL cmax = -alpha2 - beta2;
			d2[t_id] += -cmax * (d_upperbound[k + 1] + d_upperbound[k]);
		}

		__syncthreads();



		// Part 1: Forward pass once (elimination by CR)

		// Reduces it to half size of the system, and we update uneven (a2) functions
		// d_a1[i]   * x[2i-1] + d_b1[i]   * x[2i]   + d_c1[i]   * x[2i+1] = d1[i]
		// a2[i]     * x[2i]   + b2[i]     * x[2i+1] + c2[i]     * x[2i+2] = d2[i]
		// d_a1[i+1] * x[2i+1] + d_b1[i+1] * x[2i+2] + d_c1[i+1] * x[2i+3] = d1[i+1]


		REAL r_up = a2[t_id] / (d_b1[t_id + b_id * blockDim.x]);
		REAL r_down = 0.0;
		if (t_id == blockDim.x - 1)
		{
			b2[t_id] = b2[t_id] - d_c1[t_id + b_id * blockDim.x] * r_up;
			d2[t_id] = d2[t_id] - d1[t_id] * r_up;
			a2[t_id] = -d_a1[t_id + b_id * blockDim.x] * r_up;
		}
		else
		{
			r_down = c2[t_id] / (d_b1[t_id + 1 + b_id * blockDim.x]);
			b2[t_id] = b2[t_id] - d_c1[t_id + b_id * blockDim.x] * r_up - d_a1[t_id + 1 + b_id * blockDim.x] * r_down;
			d2[t_id] = d2[t_id] - d1[t_id] * r_up - d1[t_id + 1] * r_down;
			a2[t_id] = -d_a1[t_id + b_id * blockDim.x] * r_up;
			c2[t_id] = -d_c1[t_id + 1 + b_id * blockDim.x] * r_down;
		}
		__syncthreads();

		// Part 2: Solve the half even-indexed system by PCR
		int stride = 1;
		int iteration = (int)log2(REAL(blockDim.x)) - 1;
		
        // Reduction
		for (int i = 0; i < iteration; i++)
		{

			REAL r_up = 0.0; REAL r_down = 0.0;
			int down = t_id + stride;
			if (down >= blockDim.x)
			{
				down = blockDim.x - 1;
			}
			else
			{
                // down * r_down
				r_down = c2[t_id] * __rcp(b2[down]);
			}

			int up = t_id - stride;
			if (up < 0)
			{
				up = 0;
			}
			else
			{
                // up * r_down
				r_up = a2[t_id] * __rcp(b2[up]);
			}

			__syncthreads();

			REAL bNew = b2[t_id] - c2[up] * r_up - a2[down] * r_down;
			REAL dNew = d2[t_id] - d2[up] * r_up - d2[down] * r_down;
			REAL aNew = -a2[up] * r_up;
			REAL cNew = -c2[down] * r_down;

			__syncthreads();
			a2[t_id] = aNew; b2[t_id] = bNew; c2[t_id] = cNew; d2[t_id] = dNew;
			__syncthreads();
			stride *= 2;

		}
		REAL x;
        // This is blocksize / 2
		if (t_id < stride)
		{

			int i = t_id;
			int j = t_id + stride;
			REAL tmp = b2[j] * b2[i] - c2[i] * a2[j];
			x = (b2[j] * d2[i] - c2[i] * d2[j]) * __rcp(tmp);
		}
		if (t_id >= stride)
		{

			int i = t_id - stride;
			int j = t_id;
			REAL tmp = b2[j] * b2[i] - c2[i] * a2[j];
			x = (d2[j] * b2[i] - d2[i] * a2[j]) * __rcp(tmp);
		}

		d2[t_id] = x;

		__syncthreads();
		// Part 3: Back substitution in CR

		if (t_id == 0)
		{
			d1[t_id] = (d1[t_id] - d_c1[t_id + b_id * blockDim.x] * d2[t_id]) * __rcp(d_b1[t_id + b_id * blockDim.x]);
		}
		else
		{
			d1[t_id] = (d1[t_id] - d_a1[t_id + b_id * blockDim.x] * d2[t_id - 1] - d_c1[t_id + b_id * blockDim.x] * d2[t_id]) * __rcp(d_b1[t_id + b_id * blockDim.x]);
		}
		__syncthreads();

	}

    // Send results back
	d_x[2 * t_id + b_id * 2 * blockDim.x] = d1[t_id];
	d_x[2 * t_id + 1 + b_id * 2 * blockDim.x] = d2[t_id];
}

// get the price at time 0
template <typename REAL>
REAL getPrice(REAL s, REAL ds, REAL* V)
{
	int i = (int)(s / ds);
	REAL w1 = (s - ds * i) / ds;
	REAL w0 = (REAL)1.0 - w1;
	return w1 * V[i] + w0 * V[i - 1];
}