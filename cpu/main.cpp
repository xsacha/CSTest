#include <chrono>
#include "CallExample.h"
#include "FiniteDiff.h"

// Using Black-Scholes PDE

using namespace std::chrono;
struct auto_timer
{
	time_point<high_resolution_clock> startT, endT;
	bool stopped;
	auto_timer()
		: stopped(false)
	{
		start();
	}
	void start()
	{
		startT = high_resolution_clock::now();
	}
	void stop()
	{
		stopped = true;
		endT = high_resolution_clock::now();
	}

	int format()
	{
		if (!stopped) {
			endT = high_resolution_clock::now();
		}
		return (int)duration_cast<milliseconds>(endT - startT).count();
	}

	~auto_timer()
	{
		printf("Consumed %d ms\n", format());
	}
};


int main()
{
	// Set up some call option parameters
	// Example: ASX:BOQ May 30th
	// Actual BOQWW7: 0.19
	double S0 = 7.7;
	double sigma = 0.3;
	double r = 0.04;
	double T = 16.0 / 365;
	double K = 7.51;
	double s_min = 0.0, s_max = 50.0;

	CallExample option(T, r, K, sigma);
	printf("Price: %lf\n", option.UseFormula(S0));

	auto p = BlackPDE(s_min, s_max, &option);

		FiniteDiff<3000, 300> expMethod(&p);
	{
		auto_timer t;
		expMethod.ExplicitMethod();
	}

	printf("Explicit Method | Price = %lf\n", expMethod.getPrice(0.0, S0));

	// We should slices/scales that would match a future CUDA version
	FiniteDiff<200, 2048> impMethod(&p);
	{
		auto_timer t;
		impMethod.ImplicitMethod();
	}
	printf("Implicit Method | Price = %lf\n", impMethod.getPrice(0.0, S0));

	{
		auto_timer t;
		impMethod.CrankNicolsonMethod();
	}

	printf("Crank Nicolson Method | Price = %lf\n", impMethod.getPrice(0.0, S0));

	return 0;
}