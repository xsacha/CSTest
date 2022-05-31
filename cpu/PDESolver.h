#pragma once
#include "CallExample.h"

class PDESolver
{
public:
	double T, S_min, S_max;
	CallExample* option;

	PDESolver(double S_min, double S_max, CallExample* option_);
	virtual double a(double t, double s) = 0;
	virtual double b(double t, double s) = 0;
	virtual double c(double t, double s) = 0;
	virtual double d(double t, double s) = 0;
	virtual double f(double s) = 0;
	virtual double fu(double t) = 0;
	virtual double fl(double t) = 0;

};

class BlackPDE : public PDESolver
{
public:

	BlackPDE(double S_min, double S_max, CallExample* option_) :PDESolver(S_min, S_max, option_) {}
	double a(double t, double s);
	double b(double t, double s);
	double c(double t, double s);
	double d(double t, double s);
	double f(double s);
	double fl(double t);
	double fu(double t);
};
