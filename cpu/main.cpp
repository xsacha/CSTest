// TODO: A reference CPU implementation
// Using Black-Scholes PDE

#include <chrono>
#include "CallExample.h"

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
}