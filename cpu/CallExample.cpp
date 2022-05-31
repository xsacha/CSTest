#include "CallExample.h"
#include <numbers>
#include <cmath>


// Cumulative standard normal distribution
double N(double x)
{
	double gamma = 0.2316419;
	double a1 = 0.319381530;
	double a2 = -0.356563782;
	double a3 = 1.781477937;
	double a4 = -1.821255978;
	double a5 = 1.330274429;
	constexpr double sqrt_2pi_inv = std::numbers::inv_sqrtpi_v<double> / std::numbers::sqrt2_v<double>;
	double k = 1.0 / (1.0 + gamma * std::abs(x));

	double cnd = ((((a5 * k + a4) * k + a3) * k + a2) * k + a1) * k * exp(-x * x / 2.0) * sqrt_2pi_inv;
	if (x > 0.0)
	{
		return 1.0 - cnd;
	}
	else return cnd;
}

CallExample::CallExample(double T, double r, double K, double sigma)
	: T(T), r(r), K(K), sigma(sigma)
{
}

double CallExample::payOff(double s)
{
	if (s > K)
		return s - K;
	return 0.0;

}
double CallExample::UseFormula(double S0)
{
	double dPlus = (log(S0 / K) + (r + 0.5 * pow(sigma, 2.0)) * T) / (sigma * sqrt(T));
	double dMinus = dPlus - sigma * sqrt(T);
	return S0 * N(dPlus) - K * exp(-r * T) * N(dMinus);
}
