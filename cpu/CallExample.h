#pragma once

class CallExample
{
public:
	double T;
	double r;
	double K;
	double sigma;
	CallExample(double T, double r, double K, double sigma);
	double payOff(double s);
	double UseFormula(double S0);
};
