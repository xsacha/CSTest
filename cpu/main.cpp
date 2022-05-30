// TODO: A reference CPU implementation
// Using Black-Scholes PDE
// dC/dt + 0.5s^2+S^2(d^2C/dS^2) + rS(dC/dS) - rC = 0
// Initial condition: C(S, T) = max(St - K, 0)
// boundary condition: C(0, t) = 0; C(S, t) tends to S as S to infinity
// C(S, t) is defined over 0 < S < infinity and 0 <= t <= T
// but we will use an Smax for finite differences
// we price for t = 0 assuming exercise at T (European)
// We arrive at:
// du/dr = d^2u/dx^2 + (k - 1)(du/dx) - ku

#include <chrono>

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