#pragma once
#include <vector>
#include <array>
#include "PDESolver.h"

// Reference Finite Difference method
// IMAX = time slices, JMAX = asset price slices
template <size_t IMAX, size_t JMAX>
class FiniteDiff
{
  public:
    using VecD = std::vector<double>;
    // Solving class
    PDESolver* pde;

    double ds, dt;
    
    // Time given different asset prices
    std::array<std::vector<double>, IMAX + 1> V;

    FiniteDiff(PDESolver *in)
    : pde(in)
    , ds((in->S_max - in->S_min) / JMAX)
    , dt(in->T / IMAX)
    {
        for (int i = 0; i < IMAX + 1; ++i)
        {
            V[i].resize(JMAX + 1);
        }
    }

    void ExplicitMethod()
    {
        for (int j = 0; j <= JMAX; ++j)
        {
            V[IMAX][j] = f(j);
        }
        for (int i = IMAX; i > 0; --i)
        {
            V[i-1][0] = fl(i-1);
            V[i-1][JMAX] = fu(i-1);
            for (int j = 1; j< JMAX; ++j)
            {
                double A = dt*(b(i,j)*0.5-a(i,j) / ds) / ds;
                double B = 1.0-dt*c(i,j)+2.0*dt*a(i,j) / (ds * ds);
                double C = -dt*(b(i,j)*0.5+a(i,j) / ds) / ds;
                double D = -dt*d(i,j);
                V[i-1][j] = A*V[i][j-1]+B*V[i][j]+C*V[i][j+1]+D;
            }
        }
    }

    void ImplicitMethod()
    {
        VecD A(JMAX);
        VecD B(JMAX);
        VecD C(JMAX);

        for (int j = 0; j <= JMAX; ++j)
        {
            V[IMAX][j]=f(j);
        }
        for (int i = IMAX; i > 0; --i)
        {
            for(int j = 1; j < JMAX; ++j)
            {
                // Break down terms (yes, last term is redundant as a vector)
                A[j] = dt*(-b(i,j)/2.0 + a(i,j)/ds)/ds;
                B[j] = 1.0 + dt*c(i,j) - 2.0*dt*a(i,j) / (ds*ds);
                C[j] = dt*(b(i,j)/2.0 + a(i,j)/ds)/ds;
                V[i][j] -= dt * d(i,j);
            }
            V[i][1] += -A[1] * fl(i-1);
            V[i][JMAX-1] += -C[JMAX-1] * fu(i-1);
               
        
            V[i-1] = LUDecomposition(V[i], A, B, C);
        }
    }

    void CrankNicolsonMethod()
    {
        VecD A(JMAX);
        VecD B(JMAX);
        VecD C(JMAX);
        VecD E(JMAX);
        VecD F(JMAX);
        VecD G(JMAX);
        VecD q(JMAX);
	    for (int j = 0; j <= JMAX; ++j)
        {
            V[IMAX][j] = f(j);
        }
        for (int i = IMAX; i > 0; --i)
        {
            for(int j = 1; j < JMAX; ++j)
            {
                A[j]=dt*(b(i-0.5,j)*0.5-a(i-0.5,j)/ds)*0.5/ds;
                B[j]=1.0+dt*(a(i-0.5,j)/ds/ds-c(i-0.5,j)*0.5);
                C[j]=-dt*(b(i-0.5,j)*0.5+a(i-0.5,j)/ds)*0.5/ds;

                E[j]=-A[j];
                F[j]=2-B[j];                                                                                          
                G[j]=-C[j];	
                q[j]=A[j]*V[i][j-1]+B[j]*V[i][j]+C[j]*V[i][j+1] - dt*d(i,j);
            }
            q[1]+=A[1]*fl(i)-E[1]*fl(i-1);
            q[JMAX-1]+=C[JMAX-1]*fu(i)-G[JMAX-1]*fu(i-1);
            V[i-1]=LUDecomposition(q,E,F,G);
        }    
    }
    
    double getPrice(double t,double s)
    {
        int i = (int)(t/dt);
        int j = (int)((s-pde->S_min)/ds);
        double l1 = (t-dt*i)/dt, l0 = 1.0-l1;
        double w1 = (s-pde->S_min-ds*j)/ds, w0 = 1.0-w1;
        return l1*w1*V[i+1][j+1] + l1*w0*V[i+1][j]+l0*w1*V[ i ][j+1] + l0*w0*V[i][j];
    }
  private:
    VecD LUDecomposition(VecD q, VecD A, VecD B, VecD C)
    {
        VecD p(JMAX+1);
        VecD r(JMAX);
        VecD y(JMAX);
        r[1] = 1/B[1];
        y[1] = q[1]*r[1];
        C[1]=C[1]*r[1];
        for (int j = 2; j < JMAX; j++)
        {
            r[j] = 1/(B[j]-A[j]*C[j-1]);
            C[j] = C[j]*r[j];
            y[j] = (q[j]-A[j]*y[j-1])*r[j];
        }
        p[JMAX-1]=y[JMAX-1];
        for (int j=JMAX-2; j>0; j--)
        {
            p[j]=y[j]-C[j]*p[j+1];
        }
        return p;
    }
    
    double t (double i) {return dt*i;}
    double s (int j) {return pde->S_min+ds*j;}
    
    double a (double i,int j) {return pde->a(t(i),s(j));}
    double b (double i,int j) {return pde->b(t(i),s(j));}
    double c (double i,int j) {return pde->c(t(i),s(j));}
    double d (double i,int j) {return pde->d(t(i),s(j));}
    

    double f (int j) {return pde->f(s(j));}
    double fu (int i) {return pde->fu(t(i));}
    double fl (int i) {return pde->fl(t(i));}
};
