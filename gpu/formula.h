#pragma once

// Black Scholes by formula, matching reference
__device__ inline double N(double x) {
    const double gamma = 0.2316419;
	const double a1 = 0.319381530;
	const double a2 = -0.356563782;
	const double a3 = 1.781477937;
	const double a4 = -1.821255978;
	const double a5 = 1.330274429;
    constexpr double sqrt_2pi_inv = 0.39894228040143267793994605993438;


	double k = (1.0 / (1.0 + gamma * abs(x)));
	double cnd = exp(-0.5 * x * x) * (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))) * sqrt_2pi_inv;

	if(x > 0)
    {
		return 1 - cnd;
    }

	return cnd;
}

__device__ inline void UseFormulaGPU(double& result, double T, double S0, double r, double K, double sigma) {
	double sqrtT = sqrt(T);
	double dPlus = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
	double dMinus = dPlus - sigma * sqrtT;

	result = S0 * N(dPlus) - K * exp(-r * T) * N(dMinus);
}

__global__ void UseFormula(double* result, double T, double S0, double r, double K, double sigma) {
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;
    // This isn't parallel because only one option
    // Typically this would be on CPU
    if (tid == 0)
    {
        UseFormulaGPU(*result, T, S0, r, K, sigma);
    }
}