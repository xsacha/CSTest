#include "PDESolver.h"
#include <cmath>

PDESolver::PDESolver(double S_min, double S_max, CallExample* option_)
	: S_min(S_min), S_max(S_max)
	, option(option_)
	, T(option_->T)
{
}

double BlackPDE::a(double t, double s)
{
	return -0.5 * pow(option->sigma * s, 2);
}

double BlackPDE::b(double t, double s)
{
	return -option->r * s;
}

double BlackPDE::c(double t, double s)
{
	return option->r;
}
double BlackPDE::d(double t, double s)
{
	return 0.0;
}




double BlackPDE::f(double s)
{
	return option->payOff(s);

}

double BlackPDE::fl(double t)
{
	return 0.0;
}

double BlackPDE::fu(double t)
{
	return option->payOff(S_max) * exp(-option->r * (T - t));
}
