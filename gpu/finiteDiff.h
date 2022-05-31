#pragma once
#include "optimisations.h"

// Using REAL template to try to maintain double and float options

// Explicit time marching method
template <typename REAL>
__global__ void ExplicitMethod(REAL sigma, REAL Smax, REAL K, REAL T, REAL r, REAL* d_x)
{
    // TODO: Maybe pass it in so no magic numbers? Same as num_options for this?
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
	constexpr unsigned TIMESTEPS = 200;
	REAL dT = T / TIMESTEPS;

    __shared__ double V_[steps + 2];
	__shared__ double V_minus[steps + 2];
	V_[t_id + 1] = d_x[t_id + 1];
	__syncthreads();
    // Main time marching (not parallel)
    for (int i = TIMESTEPS; i > 0; --i)
    {
        __syncthreads();
        // Payouts (0 for bottom)
        if (threadIdx.x == 0)
	    {
	    	V_minus[0] = 0.0;
            // steps is jmax for now
	    	V_minus[steps - 1] = ((Smax - K) * exp(-r * dT * i));
	    }
        REAL S = (curThread * subsize + i + 1) * dS;
		// Pull it out instead of calc twice
		REAL halfdT = (REAL)0.5 * dT;
		REAL alpha = halfdT * sigma * sigma * S * S / (dS * dS);
		REAL beta = halfdT * r * S / dS;
		REAL a = -alpha + beta;
		REAL b = 1.0f + 2 * alpha + r * dT;
		REAL c = -alpha - beta;
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