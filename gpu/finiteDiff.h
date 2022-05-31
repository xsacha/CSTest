#pragma once

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
	    	V_minus[blockDim.x - 1] = ((Smax - K) * exp(-r * dT * i));
	    }
        REAL S = (curThread * subsize + i + 1) * dS;
		// Pull it out instead of calc twice
		REAL halfdT = (REAL)0.5 * dT;
		REAL alpha = halfdT * sigma * sigma * S * S / (dS * dS);
		REAL beta = halfdT * r * S / dS;
		REAL a = -alpha + beta;
		REAL b = 1.0f + 2 * alpha + r * dT;
		REAL c = -alpha - beta;
        V_minus[t_id + 1] = a * V_[t_id] + b * V_[t_id + 1] + c*V_[t_id + 2];
        __syncthreads();
        V_[t_id + 1] = V_minus[t_id + 1];
        __syncthreads();
    }

    d_x[t_id + 1] = V_[t_id + 1];
}

// Implicit Method of Black-Scholes. Using PCR+Thomas by shuffle to solve tri-diagonal system
template <typename REAL>
__global__ void ImplicitMethod(REAL sigma, REAL Smax, REAL K, REAL T, REAL r, REAL* d_x, REAL* d_upperbound)
{
    int t_id = threadIdx.x;
	int b_id = blockIdx.x;
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